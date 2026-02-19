"""
SPICEBridge — генерация SPICE netlist и запуск симуляции через ngspice.

Если ngspice не установлен — возвращает mock-результаты для демонстрации UI.
Установка ngspice: sudo apt-get install ngspice
"""

from __future__ import annotations

import os
import re
import subprocess
import tempfile
import json
from dataclasses import dataclass, field
from typing import Optional


# ─── Структуры данных ─────────────────────────────────────────────────────────

@dataclass
class SimulationResult:
    success:        bool
    netlist:        str
    stdout:         str
    stderr:         str
    measurements:   dict = field(default_factory=dict)   # {"Vout": "3.2V", "Gain": "12dB"}
    warnings:       list = field(default_factory=list)
    sim_type:       str  = "op"                           # op / tran / ac / dc


# ─── Модели компонентов (встроенные .model строки) ────────────────────────────

BUILT_IN_MODELS = """
* ─── BJT Models ─────────────────────────────────────────────────────────────
.model 2N2222 NPN(Is=14.34f Xti=3 Eg=1.11 Vaf=74.03 Bf=255.9 Ne=1.307
+        Ise=14.34f Ikf=.2847 Xtb=1.5 Br=6.092 Nc=2 Isc=0 Ikr=0 Rc=1
+        Cjc=7.306p Mjc=.3416 Vjc=.75 Fc=.5 Cje=22.01p Mje=.377 Vje=.75
+        Tr=46.91n Tf=411.1p Itf=.6 Vtf=1.7 Xtf=3 Rb=10)
.model 2N2907 PNP(Is=650.6e-18 Xti=3 Eg=1.11 Vaf=115.7 Bf=231.7 Ne=1.829
+        Ise=54.81f Ikf=1.079 Xtb=1.5 Br=3.563 Nc=2 Isc=0 Ikr=0 Rc=0
+        Cjc=14.76p Mjc=.5383 Vjc=.75 Fc=.5 Cje=19.82p Mje=.3357 Vje=.75
+        Tr=111.3n Tf=603.7p Itf=.65 Vtf=5 Xtf=1.7 Rb=10)
.model BC547 NPN(Is=7.59e-15 Bf=340 Br=6.1 Rb=0.01 Rc=0.3 Re=0.3
+        Vaf=120 Ikf=0.1 Ise=7.59e-15 Cje=11p Cjc=4p Tf=0.3n)

* ─── MOSFET Models ──────────────────────────────────────────────────────────
.model IRF540 NMOS(Level=3 Gamma=0 Delta=0 Eta=0 Theta=0 Kappa=0.2
+        Vmax=0 Xj=0 Tox=100n Uo=650 Phi=0.6 Rs=1.624m Kp=20.53u
+        W=1 L=2u Vto=3.4 Rd=1.031m Rds=444.4K Cbd=2n Pb=0.8 Mj=0.5
+        Cgso=9.027n Cgdo=9.027n Rg=1 Is=194p N=1 Tt=640n)

* ─── Diode Models ────────────────────────────────────────────────────────────
.model 1N4148 D(Is=2.52n Rs=0.568 N=1.752 Cjo=4p M=0.4 tt=20n Bv=100 Ibv=0.1u)
.model 1N4001 D(Is=14.11n N=1.984 Rs=33.89m Ikf=94.81 Xti=3 Eg=1.11 Cjo=25.89p
+             M=0.4431 Vj=0.3245 Fc=0.5 Bv=50 Ibv=10u Tt=5.771u)
.model BAT54 D(Is=2.5u N=1.05 Rs=8 Cjo=10p M=0.4 tt=1n Bv=30 Ibv=1u Vj=0.4)
.model LED D(Is=1.2e-20 Rs=0 N=2 Bv=4)
"""


# ─── SPICE значения ───────────────────────────────────────────────────────────

def normalize_value(val: str, comp_type: str) -> str:
    """Конвертирует человекочитаемые значения в SPICE формат."""
    if not val or val == "unknown":
        defaults = {
            "resistor": "1k", "capacitor": "100n", "inductor": "10u",
            "diode": "1N4148", "transistor_npn": "2N2222", "transistor_pnp": "2N2907",
            "led": "LED", "mosfet_n": "IRF540",
        }
        return defaults.get(comp_type, "1k")

    # Замена символов для SPICE
    val = (val
        .replace("Ω", "").replace("ohm", "").replace("Ohm", "")
        .replace("µ", "u").replace("μ", "u")
        .replace("kΩ", "k").replace("MΩ", "meg")
        .replace("kHz", "k").replace("MHz", "meg").replace("GHz", "g")
        .replace("mA", "m").replace("µA", "u").replace("nA", "n")
        .strip()
    )
    return val


def comp_type_to_spice_prefix(comp_type: str) -> str:
    prefixes = {
        "resistor":       "R",
        "capacitor":      "C",
        "inductor":       "L",
        "transistor_npn": "Q",
        "transistor_pnp": "Q",
        "mosfet_n":       "M",
        "mosfet_p":       "M",
        "diode":          "D",
        "zener_diode":    "D",
        "led":            "D",
        "voltage_source": "V",
        "current_source": "I",
    }
    return prefixes.get(comp_type, "X")


def comp_id_to_spice(comp_id: str, comp_type: str) -> str:
    """R1 → R1, Q1 → Q1, но если тип не совпадает с prefix — пересоздаём."""
    prefix = comp_type_to_spice_prefix(comp_type)
    num    = re.sub(r"[^0-9]", "", comp_id) or "1"
    return f"{prefix}{num}"


def default_model(comp_type: str, value: str) -> str:
    models = {
        "transistor_npn": "2N2222",
        "transistor_pnp": "2N2907",
        "mosfet_n":       "IRF540",
        "diode":          "1N4148",
        "zener_diode":    "1N4148",   # упрощение
        "led":            "LED",
    }
    # Если в value уже указан model — используем его
    known_models = ["2N2222", "2N2907", "BC547", "IRF540", "1N4148", "1N4001", "BAT54"]
    for m in known_models:
        if m.lower() in (value or "").lower():
            return m
    return models.get(comp_type, "")


# ─── Генератор netlist ────────────────────────────────────────────────────────

class NetlistGenerator:
    """
    Генерирует SPICE netlist из списка компонентов.

    Топология: линейная цепь (каждый компонент подключается последовательно),
    что достаточно для базовых расчётов. Для реальных схем нужен парсер
    нетлиста из схемного редактора (KiCad → SPICE export).
    """

    def __init__(self):
        self.node_counter = 0
        self.nodes: dict[str, tuple[str, str]] = {}   # comp_id → (n+, n-)

    def _next_node(self) -> str:
        self.node_counter += 1
        return str(self.node_counter)

    def generate(
        self,
        components: list,
        circuit_title: str = "AI Schematic Agent",
        vcc: float = 5.0,
        sim_type: str = "op",
    ) -> str:
        self.node_counter = 0
        self.nodes = {}

        lines: list[str] = []

        # Заголовок
        lines += [
            f"* {circuit_title}",
            "* Auto-generated SPICE netlist by AI Schematic Agent",
            "",
            "* ─── Power Supply ────────────────────────────────────────",
            f"Vcc vcc 0 DC {vcc}",
            "",
            "* ─── Components ─────────────────────────────────────────",
        ]

        prev_node = "vcc"  # начинаем от питания

        for comp in components:
            cid   = comp.get("id", "X1")
            ctype = comp.get("type", "resistor")
            val   = normalize_value(comp.get("value", ""), ctype)
            spice_id = comp_id_to_spice(cid, ctype)

            n_plus  = prev_node
            n_minus = self._next_node()

            self.nodes[cid] = (n_plus, n_minus)

            if ctype == "resistor":
                lines.append(f"{spice_id:6s} {n_plus:>4} {n_minus:>4}  {val}")

            elif ctype == "capacitor":
                lines.append(f"{spice_id:6s} {n_plus:>4} {n_minus:>4}  {val}  IC=0")

            elif ctype == "inductor":
                lines.append(f"{spice_id:6s} {n_plus:>4} {n_minus:>4}  {val}")

            elif ctype in ("transistor_npn", "transistor_pnp"):
                n_base    = self._next_node()
                n_emitter = self._next_node()
                model     = default_model(ctype, val)
                lines.append(f"* {cid}: collector={n_plus} base={n_base} emitter={n_emitter}")
                lines.append(f"{spice_id:6s} {n_plus:>4} {n_base:>4} {n_emitter:>4}  {model}")
                n_minus = n_emitter

            elif ctype in ("mosfet_n", "mosfet_p"):
                n_gate   = self._next_node()
                n_source = self._next_node()
                model    = default_model(ctype, val)
                lines.append(f"* {cid}: drain={n_plus} gate={n_gate} source={n_source}")
                lines.append(f"{spice_id:6s} {n_plus:>4} {n_gate:>4} {n_source:>4} {n_source:>4}  {model}")
                n_minus = n_source

            elif ctype in ("diode", "zener_diode", "led"):
                model = default_model(ctype, val)
                lines.append(f"{spice_id:6s} {n_plus:>4} {n_minus:>4}  {model}")

            elif ctype in ("voltage_source", "power"):
                lines.append(f"{spice_id:6s} {n_plus:>4} {n_minus:>4}  DC {val}")

            elif ctype == "current_source":
                lines.append(f"I{spice_id[1:]:5s} {n_plus:>4} {n_minus:>4}  DC {val}")

            elif ctype == "ground":
                n_minus = "0"
                lines.append(f"* GND connection at node {n_plus}")
                lines.append(f"Rgnd_{cid} {n_plus:>4} 0  0.001")

            else:
                lines.append(f"* {cid} ({ctype} = {val}) — skipped, manual netlist needed")
                n_minus = prev_node  # не двигаемся по цепи

            prev_node = n_minus

        # Замыкаем на землю
        lines += [
            "",
            f"* ─── Close circuit to GND ──────────────────────────",
            f"Rload {prev_node:>4}   0  10k",
            "",
        ]

        # Модели
        lines += ["", BUILT_IN_MODELS, ""]

        # Директивы симуляции
        lines += self._sim_directives(sim_type)
        lines.append(".end")

        return "\n".join(lines)

    def _sim_directives(self, sim_type: str) -> list[str]:
        directives = {
            "op":   [".op", ".save all"],
            "tran": [".tran 1us 5ms 0 0.1us", ".save all"],
            "ac":   [".ac dec 100 1Hz 10MEGHz", ".save all"],
            "dc":   [".dc Vcc 0 12 0.1", ".save all"],
        }
        return directives.get(sim_type, [".op"]) + [
            f".measure TRAN Vout_max MAX v({self._last_output_node()})",
            f".measure OP   Vout    FIND v({self._last_output_node()}) AT=0",
        ]

    def _last_output_node(self) -> str:
        return str(max(1, self.node_counter))


# ─── Парсер результатов ngspice ───────────────────────────────────────────────

class SPICEOutputParser:

    def parse(self, stdout: str, sim_type: str) -> dict:
        measurements = {}

        if sim_type == "op":
            # Ищем DC operating point: "v(1)  = 3.24"
            for m in re.finditer(r'v\((\w+)\)\s*=\s*([\d.eE+\-]+)', stdout, re.IGNORECASE):
                measurements[f"V(node{m.group(1)})"] = f"{float(m.group(2)):.4f} V"
            for m in re.finditer(r'i\((\w+)\)\s*=\s*([\d.eE+\-]+)', stdout, re.IGNORECASE):
                measurements[f"I({m.group(1)})"] = f"{float(m.group(2))*1000:.4f} mA"

        # Ищем .measure результаты
        for m in re.finditer(r'(\w+)\s*=\s*([\d.eE+\-]+)\s*(v|a|w|hz)?', stdout, re.IGNORECASE):
            name, val, unit = m.group(1), m.group(2), m.group(3) or ""
            if name.lower() not in ("tran", "dc", "ac", "op"):
                try:
                    measurements[name] = f"{float(val):.4f} {unit.upper()}"
                except ValueError:
                    pass

        # Ищем ошибки
        warnings = []
        for line in stdout.splitlines():
            if any(kw in line.lower() for kw in ["error", "warning", "singular", "failed"]):
                warnings.append(line.strip())

        return {"measurements": measurements, "warnings": warnings}


# ─── Основной класс ───────────────────────────────────────────────────────────

class SPICEBridge:
    """
    Полный пайплайн: компоненты → netlist → симуляция → результаты.

    Использование:
        bridge = SPICEBridge()
        result = bridge.simulate(components, sim_type="op", vcc=5.0)
        print(result.measurements)
        print(result.netlist)
    """

    def __init__(self, ngspice_path: str = "ngspice"):
        self.ngspice_path = ngspice_path
        self.generator    = NetlistGenerator()
        self.parser       = SPICEOutputParser()
        self._ngspice_ok  = self._check_ngspice()

    def _check_ngspice(self) -> bool:
        try:
            r = subprocess.run([self.ngspice_path, "--version"],
                               capture_output=True, timeout=5)
            return r.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    @property
    def ngspice_available(self) -> bool:
        return self._ngspice_ok

    def generate_netlist(
        self,
        components:    list,
        circuit_title: str  = "Schematic",
        vcc:           float = 5.0,
        sim_type:      str  = "op",
    ) -> str:
        return self.generator.generate(components, circuit_title, vcc, sim_type)

    def simulate(
        self,
        components:    list,
        circuit_title: str  = "Schematic",
        vcc:           float = 5.0,
        sim_type:      str  = "op",
    ) -> SimulationResult:
        netlist = self.generate_netlist(components, circuit_title, vcc, sim_type)

        if not self._ngspice_ok:
            return self._mock_simulation(netlist, sim_type)

        return self._run_ngspice(netlist, sim_type)

    def _run_ngspice(self, netlist: str, sim_type: str) -> SimulationResult:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".cir", delete=False) as f:
            f.write(netlist)
            netlist_file = f.name

        try:
            result = subprocess.run(
                [self.ngspice_path, "-b", "-o", "/dev/stdout", netlist_file],
                capture_output=True, text=True, timeout=30
            )
            parsed = self.parser.parse(result.stdout, sim_type)
            return SimulationResult(
                success=      result.returncode == 0,
                netlist=      netlist,
                stdout=       result.stdout,
                stderr=       result.stderr,
                measurements= parsed["measurements"],
                warnings=     parsed["warnings"],
                sim_type=     sim_type,
            )
        except subprocess.TimeoutExpired:
            return SimulationResult(
                success=False, netlist=netlist, stdout="", stderr="Timeout",
                warnings=["Simulation timed out after 30s"], sim_type=sim_type
            )
        finally:
            os.unlink(netlist_file)

    def _mock_simulation(self, netlist: str, sim_type: str) -> SimulationResult:
        """Реалистичные mock-результаты когда ngspice не установлен."""
        mock_op = {
            "V(node1)": "4.9823 V",
            "V(node2)": "3.2145 V",
            "V(node3)": "2.1876 V",
            "V(node4)": "0.6832 V",
            "I(Vcc)":   "-12.3456 mA",
            "I(R1)":    "0.4921 mA",
        }
        mock_tran = {
            "Vout_max":  "4.8241 V",
            "Vout_min":  "0.1823 V",
            "Period":    "0.001000 S",
            "Freq":      "1000.000 Hz",
        }
        mock_ac = {
            "Gain_1kHz":  "12.34 dB",
            "Phase_1kHz": "-45.23 °",
            "f_3dB":      "15823.45 Hz",
            "BW":         "15823.45 Hz",
        }
        measurements = {"op": mock_op, "tran": mock_tran, "ac": mock_ac}.get(sim_type, mock_op)
        return SimulationResult(
            success=      True,
            netlist=      netlist,
            stdout=       "ngspice not installed — mock results",
            stderr=       "",
            measurements= measurements,
            warnings=     ["⚠️ ngspice not found. Install: sudo apt-get install ngspice"],
            sim_type=     sim_type,
        )
