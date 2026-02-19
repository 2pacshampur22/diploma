"""
ComponentAdvisor — подбор реальных компонентов с даташитами.

Логика работы:
  1. Получает список компонентов из детектора (или от пользователя)
  2. Для каждого компонента формирует запрос к Qwen (через LLM)
  3. LLM возвращает список рекомендованных part numbers с обоснованием
  4. Верифицирует part numbers через Octopart API (если ключ есть)
  5. Возвращает структурированные рекомендации с ссылками на даташиты

Без API-ключей работает в offline-режиме: LLM предлагает компоненты
из своих знаний, ссылки генерируются по шаблону.
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass, field, asdict
from typing import Optional


# ─── Структуры данных ─────────────────────────────────────────────────────────

@dataclass
class ComponentAlternative:
    part_number:   str
    manufacturer:  str
    description:   str
    why_better:    str
    key_specs:     dict          # {"Vmax": "100V", "Imax": "500mA", ...}
    datasheet_url: str
    buy_url:       str           # Digikey / Mouser / LCSC
    price_usd:     Optional[float] = None
    in_stock:      Optional[bool]  = None
    verified:      bool            = False   # True если проверено через API


@dataclass
class ComponentSuggestion:
    component_id:   str
    component_type: str
    current_value:  str
    issue:          str                        # почему нужна замена
    alternatives:   list[ComponentAlternative] = field(default_factory=list)


# ─── Промпты для LLM ──────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert electronics engineer with deep knowledge of
electronic components, their datasheets, and real-world part numbers.
Always respond with valid JSON only. No markdown, no explanation outside JSON."""

SUGGESTION_PROMPT = """
Analyze this electronic component and suggest 3 real alternatives from major manufacturers.

Component info:
- ID: {component_id}
- Type: {component_type}
- Current value/model: {current_value}
- Circuit type: {circuit_type}
- Operating conditions: Vcc={vcc}V, I_max={imax}mA, T_ambient={temp}°C
- Issue / reason for replacement: {issue}

Requirements:
- Use only real, orderable part numbers from: TI, Vishay, Murata, Yageo, Samsung, LCSC, Bourns, Panasonic, Infineon, ON Semi, STMicro, Microchip
- Order by: best fit first
- For each alternative provide datasheet URL from official manufacturer site or datasheet.live

Return JSON array:
[
  {{
    "part_number":   "exact part number",
    "manufacturer":  "manufacturer name",
    "description":   "one line description",
    "why_better":    "specific reason this is better than current",
    "key_specs": {{
      "spec_name": "value with unit"
    }},
    "datasheet_url": "https://...",
    "buy_url":       "https://www.lcsc.com/search?q=PARTNUMBER or digikey url",
    "price_usd":     0.05
  }}
]
"""

CIRCUIT_ANALYSIS_PROMPT = """
Analyze this list of components from an electronic schematic and identify:
1. Potential issues (wrong values, incompatible types, missing bypass caps, etc.)
2. Which components most need replacement or improvement

Components: {components_json}
Circuit type: {circuit_type}

Return JSON:
{{
  "circuit_analysis": "brief description of what this circuit does",
  "overall_health": "good/fair/poor",
  "issues": [
    {{
      "component_id": "R1",
      "severity":     "high/medium/low",
      "issue":        "description of the problem",
      "suggestion":   "brief fix recommendation"
    }}
  ],
  "priority_order": ["R1", "C2", "Q1"]
}}
"""


# ─── LLM клиент ───────────────────────────────────────────────────────────────

class LLMClient:
    """
    Абстракция над LLM бэкендом.
    Поддерживает: локальный Qwen (через transformers) и OpenAI-совместимые API.
    """

    def __init__(self, backend: str = "mock", model_path: str = "", api_url: str = "", api_key: str = ""):
        """
        backend: "mock" | "local_qwen" | "openai_compatible"
        """
        self.backend    = backend
        self.model_path = model_path
        self.api_url    = api_url
        self.api_key    = api_key
        self._model     = None
        self._processor = None

        if backend == "local_qwen":
            self._load_qwen()

    def _load_qwen(self):
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            print(f"🔄 Loading Qwen for advisor: {self.model_path}")
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self._model     = AutoModelForCausalLM.from_pretrained(
                self.model_path, device_map="auto", torch_dtype="auto"
            )
            print("✅ Qwen loaded")
        except ImportError:
            print("⚠️  transformers not installed, falling back to mock")
            self.backend = "mock"

    def chat(self, system: str, user: str, max_tokens: int = 2048) -> str:
        if self.backend == "mock":
            return self._mock_response(user)
        elif self.backend == "local_qwen":
            return self._qwen_chat(system, user, max_tokens)
        elif self.backend == "openai_compatible":
            return self._api_chat(system, user, max_tokens)
        return "{}"

    def _qwen_chat(self, system: str, user: str, max_tokens: int) -> str:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        text = self._tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self._tokenizer([text], return_tensors="pt").to(self._model.device)
        output = self._model.generate(**inputs, max_new_tokens=max_tokens, temperature=0.2, do_sample=True)
        trimmed = output[0][inputs.input_ids.shape[1]:]
        return self._tokenizer.decode(trimmed, skip_special_tokens=True)

    def _api_chat(self, system: str, user: str, max_tokens: int) -> str:
        import json, urllib.request
        payload = json.dumps({
            "model": "qwen2.5-7b-instruct",
            "messages": [
                {"role": "system",  "content": system},
                {"role": "user",    "content": user},
            ],
            "max_tokens": max_tokens,
            "temperature": 0.2,
        }).encode()
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req  = urllib.request.Request(self.api_url, data=payload, headers=headers)
        resp = urllib.request.urlopen(req, timeout=60)
        data = json.loads(resp.read())
        return data["choices"][0]["message"]["content"]

    def _mock_response(self, prompt: str) -> str:
        """Реалистичные mock-ответы для тестирования без GPU."""

        if "circuit_analysis" in prompt.lower() or "issues" in prompt.lower():
            return json.dumps({
                "circuit_analysis": "Common-emitter NPN transistor amplifier with voltage divider bias",
                "overall_health": "fair",
                "issues": [
                    {"component_id": "R1", "severity": "medium",
                     "issue": "10kΩ bias resistor may be too high for stable Q-point at low hFE",
                     "suggestion": "Consider 4.7kΩ–6.8kΩ for more stable bias"},
                    {"component_id": "C1", "severity": "low",
                     "issue": "100nF bypass cap may be insufficient for low-frequency gain",
                     "suggestion": "Increase to 10µF electrolytic for full AF gain"},
                    {"component_id": "Q1", "severity": "low",
                     "issue": "BC547 has moderate hFE spread (110–800), consider fixed-gain variant",
                     "suggestion": "BC547C or 2N3904 for tighter tolerance"},
                ],
                "priority_order": ["R1", "C1", "Q1"]
            })

        # Определяем тип компонента из промпта
        comp_type = "resistor"
        for t in ["resistor","capacitor","transistor","diode","inductor","mosfet"]:
            if t in prompt.lower():
                comp_type = t
                break

        MOCK_DB = {
            "resistor": [
                {"part_number": "RC0402FR-0710KL", "manufacturer": "Yageo",
                 "description": "10kΩ 1% 0.0625W 0402 thick film resistor",
                 "why_better": "AEC-Q200 qualified, ±1% tolerance vs ±5%, tighter tempco 100ppm/°C",
                 "key_specs": {"Resistance": "10kΩ", "Tolerance": "±1%", "Power": "62.5mW",
                               "Package": "0402", "Tempco": "100ppm/°C"},
                 "datasheet_url": "https://www.yageo.com/upload/media/product/products/datasheet/rchip/PYu-RC_Group_51_RoHS_L_12.pdf",
                 "buy_url": "https://www.lcsc.com/search?q=RC0402FR-0710KL", "price_usd": 0.02},
                {"part_number": "CRCW040210K0FKED", "manufacturer": "Vishay",
                 "description": "10kΩ 1% 0.063W 0402 Dale thick film",
                 "why_better": "Dale series — proven reliability, excellent noise performance",
                 "key_specs": {"Resistance": "10kΩ", "Tolerance": "±1%", "Power": "63mW",
                               "Package": "0402", "Tempco": "100ppm/°C"},
                 "datasheet_url": "https://www.vishay.com/docs/20035/dcrcwe3.pdf",
                 "buy_url": "https://www.digikey.com/en/products/detail/vishay-dale/CRCW040210K0FKED",
                 "price_usd": 0.03},
                {"part_number": "ERA-2AEB103X", "manufacturer": "Panasonic",
                 "description": "10kΩ 0.1% 0402 thin film precision resistor",
                 "why_better": "Thin film: ±0.1% tolerance, 25ppm/°C — ideal for precision circuits",
                 "key_specs": {"Resistance": "10kΩ", "Tolerance": "±0.1%", "Power": "62.5mW",
                               "Package": "0402", "Tempco": "25ppm/°C"},
                 "datasheet_url": "https://industrial.panasonic.com/cdbs/www-data/pdf/RDO0000/AOA0000C304.pdf",
                 "buy_url": "https://www.mouser.com/ProductDetail/Panasonic/ERA-2AEB103X",
                 "price_usd": 0.15},
            ],
            "capacitor": [
                {"part_number": "GCM155R71H104KE02D", "manufacturer": "Murata",
                 "description": "100nF X7R 50V 0402 MLCC",
                 "why_better": "X7R dielectric: stable capacitance over -55°C to +125°C, ±10%",
                 "key_specs": {"Capacitance": "100nF", "Voltage": "50V", "Dielectric": "X7R",
                               "Package": "0402", "Tolerance": "±10%"},
                 "datasheet_url": "https://www.murata.com/en-us/api/pdfdownloadapi?cate=&filename=c02e.pdf",
                 "buy_url": "https://www.lcsc.com/search?q=GCM155R71H104KE02D", "price_usd": 0.03},
                {"part_number": "CL05B104KO5NNNC", "manufacturer": "Samsung",
                 "description": "100nF X5R 10V 0402 MLCC",
                 "why_better": "Excellent price/quality for decoupling, AEC-Q200, AECQ-Grade",
                 "key_specs": {"Capacitance": "100nF", "Voltage": "10V", "Dielectric": "X5R",
                               "Package": "0402", "Tolerance": "±10%"},
                 "datasheet_url": "https://mm.digikey.com/Volume0/opasdata/d220001/medias/docus/609/CL05B104KO5NNNC_Spec.pdf",
                 "buy_url": "https://www.lcsc.com/search?q=CL05B104KO5NNNC", "price_usd": 0.01},
                {"part_number": "06033C104KAT2A", "manufacturer": "AVX",
                 "description": "100nF X7R 25V 0603 MLCC",
                 "why_better": "0603 package easier to hand-solder, better for prototyping",
                 "key_specs": {"Capacitance": "100nF", "Voltage": "25V", "Dielectric": "X7R",
                               "Package": "0603", "Tolerance": "±10%"},
                 "datasheet_url": "https://www.avx.com/docs/Catalogs/cx5r.pdf",
                 "buy_url": "https://www.digikey.com/en/products/detail/kyocera-avx/06033C104KAT2A",
                 "price_usd": 0.04},
            ],
            "transistor": [
                {"part_number": "2N3904TA", "manufacturer": "ON Semiconductor",
                 "description": "NPN general-purpose transistor, 40V, 200mA, TO-92",
                 "why_better": "Industry standard, tighter hFE spread than BC547, better documented",
                 "key_specs": {"Vce_max": "40V", "Ic_max": "200mA", "hFE": "100-300",
                               "Ft": "300MHz", "Package": "TO-92"},
                 "datasheet_url": "https://www.onsemi.com/pdf/datasheet/2n3903-d.pdf",
                 "buy_url": "https://www.lcsc.com/search?q=2N3904", "price_usd": 0.05},
                {"part_number": "MMBT3904", "manufacturer": "ON Semiconductor",
                 "description": "NPN transistor SOT-23 SMD, 40V, 200mA",
                 "why_better": "SOT-23 SMD package — same specs as 2N3904 but for modern PCB design",
                 "key_specs": {"Vce_max": "40V", "Ic_max": "200mA", "hFE": "100-300",
                               "Ft": "300MHz", "Package": "SOT-23"},
                 "datasheet_url": "https://www.onsemi.com/pdf/datasheet/mmbt3904-d.pdf",
                 "buy_url": "https://www.lcsc.com/search?q=MMBT3904", "price_usd": 0.03},
                {"part_number": "BC847C", "manufacturer": "Nexperia",
                 "description": "NPN transistor SOT-23, hFE 420-800, 45V 100mA",
                 "why_better": "hFE class C (420-800) — high gain for weak signal amplification",
                 "key_specs": {"Vce_max": "45V", "Ic_max": "100mA", "hFE": "420-800",
                               "Ft": "300MHz", "Package": "SOT-23"},
                 "datasheet_url": "https://assets.nexperia.com/documents/data-sheet/BC847_SER.pdf",
                 "buy_url": "https://www.lcsc.com/search?q=BC847C", "price_usd": 0.04},
            ],
            "diode": [
                {"part_number": "1N4148W", "manufacturer": "Vishay",
                 "description": "High-speed switching diode SOD-123, 100V 150mA",
                 "why_better": "SOD-123 SMD variant of classic 1N4148, same specs, modern package",
                 "key_specs": {"Vr_max": "100V", "If_max": "150mA", "trr": "4ns",
                               "Vf": "1V@10mA", "Package": "SOD-123"},
                 "datasheet_url": "https://www.vishay.com/docs/85748/1n4148w.pdf",
                 "buy_url": "https://www.lcsc.com/search?q=1N4148W", "price_usd": 0.02},
                {"part_number": "BAT54S", "manufacturer": "Nexperia",
                 "description": "Dual Schottky diode SOT-23, 30V 200mA",
                 "why_better": "Schottky: Vf=0.3V vs 0.7V for Si — lower drop, faster switching",
                 "key_specs": {"Vr_max": "30V", "If_max": "200mA", "Vf": "0.3V@1mA",
                               "trr": "<5ns", "Package": "SOT-23 dual"},
                 "datasheet_url": "https://assets.nexperia.com/documents/data-sheet/BAT54S.pdf",
                 "buy_url": "https://www.lcsc.com/search?q=BAT54S", "price_usd": 0.05},
                {"part_number": "SS14", "manufacturer": "Vishay",
                 "description": "Schottky rectifier SMA, 40V 1A",
                 "why_better": "1A rated Schottky for power rectification, low Vf=0.45V",
                 "key_specs": {"Vr_max": "40V", "If_avg": "1A", "Vf": "0.45V@1A",
                               "Package": "SMA (DO-214AC)"},
                 "datasheet_url": "https://www.vishay.com/docs/88746/ss12thru.pdf",
                 "buy_url": "https://www.lcsc.com/search?q=SS14", "price_usd": 0.06},
            ],
            "inductor": [
                {"part_number": "LQH31CN100K03L", "manufacturer": "Murata",
                 "description": "100µH 1210 SMD power inductor, 60mA, 17Ω",
                 "why_better": "Shielded construction, lower EMI, AEC-Q200 for automotive",
                 "key_specs": {"Inductance": "100µH", "Isat": "60mA", "DCR": "17Ω",
                               "SRF": "4MHz", "Package": "1210"},
                 "datasheet_url": "https://www.murata.com/en-us/api/pdfdownloadapi?cate=&filename=o05e.pdf",
                 "buy_url": "https://www.lcsc.com/search?q=LQH31CN100K03L", "price_usd": 0.35},
                {"part_number": "SRR1260-101Y", "manufacturer": "Bourns",
                 "description": "100µH shielded power inductor, 1.5A, 0.154Ω",
                 "why_better": "High current capability 1.5A, low DCR 154mΩ — for power circuits",
                 "key_specs": {"Inductance": "100µH", "Irated": "1.5A", "DCR": "0.154Ω",
                               "Package": "SRR1260 (12.5x12.5mm)"},
                 "datasheet_url": "https://www.bourns.com/docs/Product-Datasheets/SRR1260.pdf",
                 "buy_url": "https://www.lcsc.com/search?q=SRR1260-101Y", "price_usd": 0.80},
                {"part_number": "SWPA4020S100MT", "manufacturer": "Sunlord",
                 "description": "100µH 4020 SMD shielded inductor, 200mA",
                 "why_better": "Compact 4020 footprint, good cost/performance for DC-DC converters",
                 "key_specs": {"Inductance": "100µH", "Isat": "200mA", "DCR": "2.5Ω",
                               "Package": "4020"},
                 "datasheet_url": "https://www.lcsc.com/datasheet/lcsc_datasheet_SWPA4020S100MT.pdf",
                 "buy_url": "https://www.lcsc.com/search?q=SWPA4020S100MT", "price_usd": 0.12},
            ],
        }

        alts = MOCK_DB.get(comp_type, MOCK_DB["resistor"])
        return json.dumps(alts)


# ─── Octopart API клиент ──────────────────────────────────────────────────────

class OctopartClient:
    """
    Верификация part numbers через Octopart API v4.
    Бесплатный план: 1000 запросов/месяц.
    Получить ключ: https://octopart.com/api/home
    """

    BASE_URL = "https://octopart.com/api/v4/rest"

    def __init__(self, api_key: str = ""):
        self.api_key = api_key

    def search_part(self, part_number: str) -> dict | None:
        if not self.api_key:
            return None
        url = (f"{self.BASE_URL}/parts/search"
               f"?q={urllib.parse.quote(part_number)}"
               f"&apikey={self.api_key}&limit=1")
        try:
            req  = urllib.request.Request(url, headers={"User-Agent": "SchematicAgent/1.0"})
            resp = urllib.request.urlopen(req, timeout=10)
            data = json.loads(resp.read())
            results = data.get("results", [])
            if results:
                item = results[0]["item"]
                return {
                    "mpn":          item.get("mpn"),
                    "manufacturer": item.get("manufacturer", {}).get("name"),
                    "datasheet":    self._extract_datasheet(item),
                    "in_stock":     self._check_stock(item),
                }
        except Exception as e:
            print(f"⚠️  Octopart error for {part_number}: {e}")
        return None

    def _extract_datasheet(self, item: dict) -> str:
        for doc in item.get("document_collections", []):
            if doc.get("name") == "Datasheets":
                docs = doc.get("documents", [])
                if docs:
                    return docs[0].get("url", "")
        return ""

    def _check_stock(self, item: dict) -> bool:
        for offer in item.get("offers", []):
            if offer.get("in_stock_quantity", 0) > 0:
                return True
        return False


# ─── Генератор ссылок на покупку ─────────────────────────────────────────────

def make_buy_links(part_number: str) -> dict:
    enc = urllib.parse.quote(part_number)
    return {
        "lcsc":    f"https://www.lcsc.com/search?q={enc}",
        "digikey": f"https://www.digikey.com/en/products/filter?keywords={enc}",
        "mouser":  f"https://www.mouser.com/Search/Refine?Keyword={enc}",
    }


# ─── Основной класс ───────────────────────────────────────────────────────────

class ComponentAdvisor:
    """
    Главный класс подбора компонентов.

    Пример использования:
        advisor = ComponentAdvisor()
        analysis = advisor.analyze_circuit(components, circuit_type="amplifier")
        suggestions = advisor.suggest_alternatives(components, analysis, circuit_context)
        report = advisor.build_report(suggestions)
    """

    def __init__(
        self,
        llm_backend:    str = "mock",
        model_path:     str = "",
        api_url:        str = "",
        api_key:        str = "",
        octopart_key:   str = "",
    ):
        self.llm      = LLMClient(backend=llm_backend, model_path=model_path,
                                  api_url=api_url, api_key=api_key)
        self.octopart = OctopartClient(api_key=octopart_key)

    # ── Публичный API ─────────────────────────────────────────────────────────

    def analyze_circuit(self, components: list, circuit_type: str = "unknown") -> dict:
        """
        Запрашивает у LLM анализ всей схемы целиком.
        Возвращает dict с полями: circuit_analysis, overall_health, issues, priority_order
        """
        prompt = CIRCUIT_ANALYSIS_PROMPT.format(
            components_json=json.dumps(components, ensure_ascii=False),
            circuit_type=circuit_type,
        )
        raw = self.llm.chat(SYSTEM_PROMPT, prompt)
        return self._parse_json(raw, default={
            "circuit_analysis": "Unable to analyze",
            "overall_health":   "unknown",
            "issues":           [],
            "priority_order":   [],
        })

    def suggest_alternatives(
        self,
        components:      list,
        circuit_analysis: dict,
        circuit_context:  dict | None = None,
    ) -> list[ComponentSuggestion]:
        """
        Для каждого проблемного компонента запрашивает 3 альтернативы у LLM,
        затем опционально верифицирует через Octopart.
        """
        ctx = circuit_context or {}
        suggestions: list[ComponentSuggestion] = []

        # Определяем компоненты для анализа
        issues_map = {i["component_id"]: i for i in circuit_analysis.get("issues", [])}
        priority   = circuit_analysis.get("priority_order", [c["id"] for c in components])

        for comp in components:
            cid = comp.get("id", "?")

            issue_info = issues_map.get(cid, {})
            issue_text = issue_info.get("issue", "General review — check if optimal for this circuit")

            prompt = SUGGESTION_PROMPT.format(
                component_id=cid,
                component_type=comp.get("type", "unknown"),
                current_value=comp.get("value") or "unknown",
                circuit_type=ctx.get("circuit_type", "unknown"),
                vcc=ctx.get("vcc", "5"),
                imax=ctx.get("imax", "100"),
                temp=ctx.get("temp", "25"),
                issue=issue_text,
            )

            raw   = self.llm.chat(SYSTEM_PROMPT, prompt)
            alts_raw = self._parse_json(raw, default=[])

            alternatives = []
            for a in (alts_raw if isinstance(alts_raw, list) else []):
                # Верификация через Octopart если ключ есть
                verified_data = self.octopart.search_part(a.get("part_number", ""))
                buy_links     = make_buy_links(a.get("part_number", ""))

                alt = ComponentAlternative(
                    part_number=   a.get("part_number",  ""),
                    manufacturer=  a.get("manufacturer", ""),
                    description=   a.get("description",  ""),
                    why_better=    a.get("why_better",   ""),
                    key_specs=     a.get("key_specs",    {}),
                    datasheet_url= (verified_data or {}).get("datasheet") or a.get("datasheet_url", ""),
                    buy_url=       a.get("buy_url") or buy_links["lcsc"],
                    price_usd=     a.get("price_usd"),
                    in_stock=      (verified_data or {}).get("in_stock"),
                    verified=      verified_data is not None,
                )
                alternatives.append(alt)

            suggestions.append(ComponentSuggestion(
                component_id=   cid,
                component_type= comp.get("type", "unknown"),
                current_value=  comp.get("value") or "unknown",
                issue=          issue_text,
                alternatives=   alternatives,
            ))

        return suggestions

    def build_report(self, analysis: dict, suggestions: list[ComponentSuggestion]) -> dict:
        """
        Собирает финальный отчёт для передачи в UI или сохранения в файл.
        """
        return {
            "circuit_analysis":  analysis.get("circuit_analysis", ""),
            "overall_health":    analysis.get("overall_health", "unknown"),
            "issues_count":      len(analysis.get("issues", [])),
            "suggestions": [
                {
                    "component_id":   s.component_id,
                    "component_type": s.component_type,
                    "current_value":  s.current_value,
                    "issue":          s.issue,
                    "alternatives": [asdict(a) for a in s.alternatives],
                }
                for s in suggestions
            ],
        }

    # ── Приватные ─────────────────────────────────────────────────────────────

    @staticmethod
    def _parse_json(text: str, default):
        text = re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("`").strip()
        # Ищем первый JSON-объект или массив
        for pattern in [r'\[.*\]', r'\{.*\}']:
            m = re.search(pattern, text, re.DOTALL)
            if m:
                try:
                    return json.loads(m.group())
                except json.JSONDecodeError:
                    pass
        try:
            return json.loads(text)
        except Exception:
            return default
