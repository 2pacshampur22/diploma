"""
Schematic AI Agent — полный интерфейс
Вкладки: Analyze → Simulate → Optimize → Dataset

Запуск:
    cd schematic-agent
    pip install gradio pillow opencv-python numpy
    python ui/app.py
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import gradio as gr
from PIL import Image, ImageDraw

# ─── Пути для импорта ─────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from simulation.spice_bridge        import SPICEBridge
from optimization.component_advisor import ComponentAdvisor
from vision.component_detector      import SchematicAnalyzer

# ─── Цвета по типам ───────────────────────────────────────────────────────────
COMPONENT_COLORS = {
    "resistor":       "#FF6B35",
    "capacitor":      "#4ECDC4",
    "inductor":       "#45B7D1",
    "transistor_npn": "#96CEB4",
    "transistor_pnp": "#88D8A3",
    "mosfet_n":       "#FFEAA7",
    "mosfet_p":       "#DDA0DD",
    "diode":          "#F8A5C2",
    "zener_diode":    "#F78FB3",
    "led":            "#FFF176",
    "ic":             "#B39DDB",
    "transformer":    "#FFAB76",
    "connector":      "#90A4AE",
    "voltage_source": "#EF5350",
    "current_source": "#E53935",
    "ground":         "#78909C",
    "power":          "#FF8F00",
    "unknown":        "#BDBDBD",
}
COMPONENT_TYPES = sorted(COMPONENT_COLORS.keys())

# ─── Глобальное состояние ─────────────────────────────────────────────────────
state = {
    "image_path":    None,
    "components":    [],
    "circuit_type":  "unknown",
    "sim_result":    None,
    "advisor_report": None,
}

# Сервисы
spice_bridge = SPICEBridge()
advisor      = ComponentAdvisor(llm_backend="mock")   # заменить на "local_qwen" когда натренирую модель
analyzer = SchematicAnalyzer(
    model_name="qwen-vl-max",  # Облачная модель
    backend="qwen_agent"
)


# Вспомогательные функции

def draw_annotations(image_path: str, components: list) -> Image.Image:
    img  = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img, "RGBA")
    for comp in components:
        bbox = comp.get("bbox", [])
        if len(bbox) != 4:
            continue
        x1, y1, x2, y2 = bbox
        hex_c = COMPONENT_COLORS.get(comp.get("type", "unknown"), "#BDBDBD")
        r, g, b = int(hex_c[1:3],16), int(hex_c[3:5],16), int(hex_c[5:7],16)
        draw.rectangle([x1,y1,x2,y2], fill=(r,g,b,55), outline=(r,g,b,230), width=2)
        label = f"{comp['id']}: {comp.get('value') or comp.get('type','?')}"
        draw.rectangle([x1, y1-18, x1+len(label)*7+8, y1], fill=(r,g,b,210))
        draw.text((x1+4, y1-16), label, fill="white")
        conf = comp.get("confidence", 0)
        draw.text((x2-32, y2-14), f"{conf:.0%}", fill=(r,g,b,210))
    return img


def _normalize_table(table_data) -> list:
    if table_data is None:
        return []
    try:
        import pandas as pd
        if isinstance(table_data, pd.DataFrame):
            return [] if table_data.empty else table_data.values.tolist()
    except ImportError:
        pass
    if isinstance(table_data, list):
        return table_data
    return []


def table_to_components(table_data) -> list:
    rows = _normalize_table(table_data)
    out  = []
    for row in rows:
        if len(row) < 5:
            continue

        def s(v, d=""):   # safe str, handle NaN
            return d if (v is None or (isinstance(v, float) and v != v)) else str(v)
        def f(v):         # safe float
            try: return float(v)
            except: return 0.0
        def b(v):         # safe bool
            if isinstance(v, bool): return v
            if isinstance(v, str):  return v.lower() in ("true","1","yes")
            try: return bool(v)
            except: return False

        bbox_str = s(row[5]) if len(row) > 5 else ""
        try:    bbox = json.loads(bbox_str) if bbox_str.strip() else []
        except: bbox = []

        out.append({
            "id":         s(row[0], "?"),
            "type":       s(row[1], "unknown"),
            "value":      s(row[2]),
            "confidence": f(row[3]),
            "verified":   b(row[4]),
            "bbox":       bbox,
        })
    return out


def components_to_table(components: list) -> list:
    return [
        [c["id"], c["type"], c.get("value",""), round(c.get("confidence",0),2),
         c.get("verified", False), json.dumps(c.get("bbox",[]))]
        for c in components
    ]


def save_annotation(image_path: str, components: list) -> str:
    save_dir = "data/annotated"
    os.makedirs(save_dir, exist_ok=True)
    stem = Path(image_path).stem
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    ann  = {"image": Path(image_path).name, "timestamp": ts,
            "components": components, "circuit_type": state["circuit_type"]}
    json_path = f"{save_dir}/{stem}_{ts}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(ann, f, indent=2, ensure_ascii=False)
    Image.open(image_path).save(f"{save_dir}/{stem}_{ts}.png")
    return json_path



def on_analyze(image_path):
    if not image_path:
        return None, [], "No image uploaded"
    state["image_path"] = image_path
    result = analyzer.detect_components(image_path)
    comps  = result.get("components", [])
    state["components"]   = comps
    state["circuit_type"] = result.get("circuit_type", "unknown")
    annotated = draw_annotations(image_path, comps)
    warnings  = result.get("warnings", [])
    status    = f"Found {len(comps)} components · Circuit: {state['circuit_type']}"
    if warnings:
        status += "\n" + "\n".join(warnings)
    return annotated, components_to_table(comps), status


def on_table_change(table_data):
    if not state["image_path"] or table_data is None:
        return None
    comps = table_to_components(table_data)
    state["components"] = comps
    return draw_annotations(state["image_path"], comps)


def on_add_component(cid, ctype, value, table_data):
    if not cid or not ctype:
        return _normalize_table(table_data), "Fill ID and Type"
    rows = list(_normalize_table(table_data))
    rows.append([cid, ctype, value or "", 1.0, True, "[]"])
    return rows, f"Added {cid}"


def on_delete_component(del_ids, table_data):
    if not del_ids or _normalize_table(table_data) == []:
        return table_data
    ids_to_del = {s.strip() for s in del_ids.split(",")}
    return [r for r in _normalize_table(table_data) if str(r[0]) not in ids_to_del]


def on_save_for_training(table_data):
    if not state["image_path"]:
        return "No image loaded"
    comps = [c for c in table_to_components(table_data) if c.get("verified")]
    if not comps:
        return "No verified components. Check ✓ in the table first."
    path = save_annotation(state["image_path"], comps)
    return f"Saved {len(comps)} verified components → {path}"


def on_export_json(table_data):
    comps = table_to_components(table_data)
    data  = {"components": comps, "circuit_type": state["circuit_type"],
             "generated": datetime.now().isoformat()}
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
    json.dump(data, tmp, indent=2, ensure_ascii=False)
    tmp.close()
    return tmp.name


# Simulate

def on_simulate(table_data, sim_type, vcc):
    comps = table_to_components(table_data)
    if not comps:
        return "No components. Analyze a schematic first.", "", ""

    result = spice_bridge.simulate(
        comps,
        circuit_title=f"AI Agent — {state['circuit_type']}",
        vcc=float(vcc),
        sim_type=sim_type,
    )
    state["sim_result"] = result

    # Статус
    icon   = "V" if result.success else "X"
    status = f"{icon} Simulation {'complete' if result.success else 'failed'} · Type: {sim_type.upper()}"
    if result.warnings:
        status += "\n" + "\n".join(result.warnings)

    # Измерения
    meas_lines = ["Measurements"]
    for k, v in result.measurements.items():
        meas_lines.append(f"{k:<22} = {v}")
    measurements_str = "\n".join(meas_lines) if result.measurements else "No measurements"

    return status, result.netlist, measurements_str


def on_download_netlist(table_data, sim_type, vcc):
    comps   = table_to_components(table_data)
    netlist = spice_bridge.generate_netlist(comps, vcc=float(vcc), sim_type=sim_type)
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".cir", delete=False)
    tmp.write(netlist)
    tmp.close()
    return tmp.name


# Optimize

def on_analyze_circuit(table_data):
    comps = table_to_components(table_data)
    if not comps:
        return "No components to analyze", []

    analysis = advisor.analyze_circuit(comps, circuit_type=state["circuit_type"])
    state["advisor_report"] = {"analysis": analysis}

    # Красивый вывод анализа
    health_icon = {"good": "🟢", "fair": "🟡", "poor": "🔴"}.get(analysis.get("overall_health",""), "⚪")
    lines  = [
        f"{health_icon} Circuit: {analysis.get('circuit_analysis', '—')}",
        f"   Health: {analysis.get('overall_health','?').upper()}",
        "",
        "Issues found:",
    ]
    issues_rows = []
    for issue in analysis.get("issues", []):
        sev  = issue.get("severity","?")
        icon = {"high":"🔴","medium":"🟡","low":"🔵"}.get(sev,"⚪")
        lines.append(f"  {icon} [{sev.upper()}] {issue.get('component_id','?')}: {issue.get('issue','')}")
        lines.append(f"      → {issue.get('suggestion','')}")
        issues_rows.append([issue.get("component_id","?"), sev, issue.get("issue",""),
                             issue.get("suggestion","")])

    prio = analysis.get("priority_order", [])
    if prio:
        lines += ["", f"Priority order: {' → '.join(prio)}"]

    return "\n".join(lines), issues_rows


def on_get_suggestions(table_data, vcc, imax, temp):
    comps = table_to_components(table_data)
    if not comps:
        return "No components", []

    analysis = (state.get("advisor_report") or {}).get("analysis") or \
               advisor.analyze_circuit(comps, state["circuit_type"])

    context = {
        "circuit_type": state["circuit_type"],
        "vcc":  vcc,
        "imax": imax,
        "temp": temp,
    }

    suggestions = advisor.suggest_alternatives(comps, analysis, context)
    report      = advisor.build_report(analysis, suggestions)
    state["advisor_report"] = {"analysis": analysis, "report": report}

    # Строим таблицу для UI
    rows = []
    for sug in report.get("suggestions", []):
        for alt in sug.get("alternatives", []):
            rows.append([
                sug["component_id"],
                sug["current_value"],
                alt["part_number"],
                alt["manufacturer"],
                alt["why_better"],
                f"${alt.get('price_usd','?')}",
                alt["datasheet_url"],
                alt["buy_url"],
            ])

    status = f" {len(suggestions)} components analyzed · {len(rows)} alternatives found"
    return status, rows


def on_export_report(table_data):
    report = (state.get("advisor_report") or {}).get("report")
    if not report:
        return None

    # Человекочитаемый текстовый отчёт
    lines = [
        "═" * 60,
        "  AI SCHEMATIC AGENT — OPTIMIZATION REPORT",
        f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "═" * 60,
        "",
        f"Circuit: {report.get('circuit_analysis','')}",
        f"Health:  {report.get('overall_health','').upper()}",
        f"Issues:  {report.get('issues_count',0)}",
        "",
    ]
    for sug in report.get("suggestions", []):
        lines += [
            f"┌─ {sug['component_id']} ({sug['component_type']}) = {sug['current_value']}",
            f"│  Issue: {sug['issue']}",
            "│  Alternatives:",
        ]
        for i, alt in enumerate(sug.get("alternatives", []), 1):
            specs = " | ".join(f"{k}: {v}" for k,v in alt.get("key_specs",{}).items())
            lines += [
                f"│  {i}. {alt['part_number']} ({alt['manufacturer']})",
                f"│     {alt['description']}",
                f"│     Why: {alt['why_better']}",
                f"│     Specs: {specs}",
                f"│     Price: ${alt.get('price_usd','?')} | Buy: {alt['buy_url']}",
                f"│     Datasheet: {alt['datasheet_url']}",
            ]
        lines.append("└" + "─"*50)

    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False,
                                      encoding="utf-8")
    tmp.write("\n".join(lines))
    tmp.close()
    return tmp.name


# Dataset 

def refresh_dataset():
    ann_dir = Path("data/annotated")
    if not ann_dir.exists():
        return [], "No annotations yet. Analyze and save schematics first."
    rows = []
    for jf in sorted(ann_dir.glob("*.json"), reverse=True):
        try:
            with open(jf) as f:
                d = json.load(f)
            comps    = d.get("components",[])
            verified = sum(1 for c in comps if c.get("verified"))
            rows.append([jf.name, d.get("circuit_type","?"), len(comps), verified,
                         d.get("timestamp","?")])
        except Exception:
            pass
    stats = (f"{len(rows)} schematics · "
             f"{sum(r[2] for r in rows)} components · "
             f"{sum(r[3] for r in rows)} verified")
    return rows, stats


def export_dataset_jsonl():
    ann_dir = Path("data/annotated")
    records = []
    for jf in (ann_dir.glob("*.json") if ann_dir.exists() else []):
        try:
            with open(jf) as f:
                records.append(json.load(f))
        except Exception:
            pass
    if not records:
        return None
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False, encoding="utf-8")
    for r in records:
        tmp.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.close()
    return tmp.name


# CSS
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

:root {
    --bg:       #080c10;
    --surf:     #0d1319;
    --surf2:    #131b24;
    --surf3:    #192230;
    --border:   #1e2d3d;
    --border2:  #2a3f55;
    --accent:   #00d4ff;
    --accent2:  #ff6b35;
    --accent3:  #7fff6e;
    --text:     #cdd9e5;
    --muted:    #5a7184;
    --danger:   #ff4d4d;
    --warn:     #ffb347;
}

body, .gradio-container {
    background: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
}

/* Header */
.agent-header {
    background: linear-gradient(135deg, var(--surf) 0%, #0a1929 100%);
    border: 1px solid var(--border2);
    border-radius: 10px;
    padding: 20px 28px;
    margin-bottom: 14px;
    position: relative;
    overflow: hidden;
}
.agent-header::after {
    content: '';
    position: absolute; top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, var(--accent), var(--accent2), var(--accent3));
}
.agent-header h1 {
    font-family: 'Space Mono', monospace;
    font-size: 22px; font-weight: 700; margin: 0;
    color: var(--accent);
    letter-spacing: 0.05em;
}
.agent-header p { color: var(--muted); margin: 4px 0 0; font-size: 12px; }

/* Legend */
.legend-wrap { padding: 6px 0 10px; }
.legend-wrap span {
    font-size: 10px; font-family: 'Space Mono', monospace;
    padding: 2px 7px; border-radius: 3px; margin: 2px;
    display: inline-block; letter-spacing: 0.04em;
}

/* Panels */
.gr-panel, .gr-box { background: var(--surf) !important; border: 1px solid var(--border) !important; border-radius: 8px !important; }
label, .gr-form > label, .block > label span { color: var(--muted) !important; font-size: 11px !important; letter-spacing: 0.08em !important; text-transform: uppercase !important; }

/* Tabs */
.tab-nav { border-bottom: 1px solid var(--border) !important; }
.tab-nav button { color: var(--muted) !important; font-family: 'Space Mono', monospace !important; font-size: 12px !important; border-radius: 0 !important; border-bottom: 2px solid transparent !important; transition: all 0.2s !important; }
.tab-nav button.selected, .tab-nav button:hover { color: var(--accent) !important; border-bottom-color: var(--accent) !important; }

/* Buttons */
button { border-radius: 5px !important; font-family: 'Space Mono', monospace !important; font-size: 11px !important; letter-spacing: 0.06em !important; transition: all 0.15s !important; }
button.primary   { background: var(--accent) !important; color: #000 !important; font-weight: 700 !important; border: none !important; }
button.secondary { background: var(--surf3) !important; color: var(--text) !important; border: 1px solid var(--border2) !important; }
button.success   { background: var(--accent3) !important; color: #000 !important; font-weight: 700 !important; border: none !important; }
button.warn-btn  { background: var(--accent2) !important; color: #000 !important; font-weight: 700 !important; border: none !important; }
button:hover     { filter: brightness(1.15) !important; }

/* Inpus */
input[type="text"], textarea, select, .gr-text-input {
    background: var(--surf2) !important; color: var(--text) !important;
    border: 1px solid var(--border2) !important; border-radius: 5px !important;
    font-family: 'Space Mono', monospace !important; font-size: 12px !important;
}
input:focus, textarea:focus { border-color: var(--accent) !important; outline: none !important; box-shadow: 0 0 0 2px rgba(0,212,255,0.15) !important; }

/* Code blocks */
.gr-code, code, pre { background: var(--surf2) !important; color: var(--accent3) !important; font-family: 'Space Mono', monospace !important; font-size: 11px !important; border: 1px solid var(--border) !important; }

/* Dataframe */
table { background: var(--surf) !important; color: var(--text) !important; border-color: var(--border) !important; font-size: 11px !important; }
thead tr { background: var(--surf3) !important; }
th { color: var(--accent) !important; font-family: 'Space Mono', monospace !important; font-size: 10px !important; letter-spacing: 0.1em !important; text-transform: uppercase !important; }
tr:hover { background: var(--surf3) !important; }

/* ── Status ──────────────────────────────────────────── */
.status-out textarea { border-left: 3px solid var(--accent2) !important; background: var(--surf2) !important; font-family: 'Space Mono', monospace !important; font-size: 11px !important; }

/* ── Section labels ──────────────────────────────────── */
.sec-label { color: var(--muted) !important; font-family: 'Space Mono', monospace !important; font-size: 10px !important; letter-spacing: 0.15em !important; text-transform: uppercase !important; border-bottom: 1px solid var(--border) !important; padding-bottom: 4px !important; margin-bottom: 8px !important; }
"""


# Легенда
def legend_html():
    items = "".join(
        f'<span style="background:{c};color:{"#000" if c in ["#FFEAA7","#FFF176","#88D8A3"] else "#fff"}'
        f';padding:2px 6px;border-radius:3px;margin:2px;display:inline-block">{t}</span>'
        for t, c in COMPONENT_COLORS.items()
    )
    return f'<div class="legend-wrap">{items}</div>'


# UI 
with gr.Blocks(title="⚡ Schematic AI Agent") as app:

    gr.HTML("""
    <div class="agent-header">
      <h1>⚡ SCHEMATIC AI AGENT</h1>
      <p>Qwen2.5-VL · Component Detection · SPICE Simulation · Part Optimization · Fine-tuning</p>
    </div>
    """)
    gr.HTML(legend_html())

    with gr.Tabs():

        with gr.Tab("🔍 Analyze"):

            with gr.Row():
                with gr.Column(scale=1, min_width=300):
                    gr.Markdown("##### Upload schematic", elem_classes=["sec-label"])
                    img_input   = gr.Image(label="", type="filepath", height=280)
                    analyze_btn = gr.Button("⚡ ANALYZE", variant="primary", elem_classes=["primary"])
                    status_out  = gr.Textbox(label="Status", interactive=False,
                                             lines=2, elem_classes=["status-out"])

                    gr.Markdown("##### Add component", elem_classes=["sec-label"])
                    with gr.Row():
                        new_id   = gr.Textbox(label="ID",   placeholder="R5", scale=1)
                        new_type = gr.Dropdown(COMPONENT_TYPES, label="Type", scale=2)
                    new_val    = gr.Textbox(label="Value", placeholder="10kΩ")
                    with gr.Row():
                        add_btn    = gr.Button("➕ Add",    elem_classes=["secondary"])
                        add_status = gr.Textbox(label="",  interactive=False, scale=2)

                    gr.Markdown("##### Delete components", elem_classes=["sec-label"])
                    del_ids  = gr.Textbox(label="IDs (comma-sep)", placeholder="R1,C2")
                    del_btn  = gr.Button("🗑 Delete", elem_classes=["secondary"])

                with gr.Column(scale=2):
                    gr.Markdown("##### Annotated schematic", elem_classes=["sec-label"])
                    ann_out = gr.Image(label="", height=480, interactive=False)

            gr.Markdown("##### Components table  *(double-click cell to edit)*",
                        elem_classes=["sec-label"])
            comp_table = gr.Dataframe(
                headers=["ID", "Type", "Value", "Confidence", "Verified ✓", "BBox"],
                datatype=["str","str","str","number","bool","str"],
                interactive=True, wrap=True,
            )
            with gr.Row():
                save_train_btn  = gr.Button("🎓 Save for Training", elem_classes=["success"])
                export_json_btn = gr.Button("💾 Export JSON",       elem_classes=["secondary"])
                train_status    = gr.Textbox(label="", interactive=False, scale=3)
            export_json_file = gr.File(label="JSON File", visible=False)

        with gr.Tab("⚡ Simulate"):
            gr.Markdown(
                "Генерирует SPICE netlist из компонентов на вкладке Analyze и запускает ngspice.\n\n"
                f"**ngspice status:** {'installed' if spice_bridge.ngspice_available else '⚠️ not found — mock results'}"
            )
            with gr.Row():
                sim_type    = gr.Radio(["op","tran","ac","dc"], value="op",
                                       label="Simulation type",
                                       info="op=DC point | tran=transient | ac=freq | dc=sweep")
                vcc_slider  = gr.Slider(1, 30, 5, step=0.5, label="Vcc (V)")
                sim_btn     = gr.Button(" RUN SIMULATION", variant="primary",
                                        elem_classes=["primary"], scale=1)

            sim_status   = gr.Textbox(label="Status", interactive=False,
                                      lines=3, elem_classes=["status-out"])
            with gr.Row():
                meas_out    = gr.Textbox(label="Measurements", interactive=False,
                                         lines=12, scale=1)
                netlist_out = gr.Code(label="SPICE Netlist", language="python",
                                      lines=18, scale=2)

            dl_netlist_btn  = gr.Button(" Download .cir", elem_classes=["secondary"])
            netlist_dl_file = gr.File(label="", visible=False)

        with gr.Tab(" Optimize"):
            gr.Markdown(
                "AI анализирует схему целиком и подбирает лучшие реальные компоненты "
                "с ссылками на даташиты и магазины."
            )
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("##### Circuit context", elem_classes=["sec-label"])
                    ctx_vcc  = gr.Slider(1, 50,  5,   step=0.5, label="Vcc (V)")
                    ctx_imax = gr.Slider(1, 5000, 100, step=10,  label="I_max (mA)")
                    ctx_temp = gr.Slider(-40, 125, 25, step=5,   label="T_ambient (°C)")

                    gr.Markdown("##### Actions", elem_classes=["sec-label"])
                    analyze_circuit_btn = gr.Button(" Analyze Circuit Issues",
                                                    elem_classes=["secondary"])
                    suggest_btn         = gr.Button(" GET ALTERNATIVES",
                                                    variant="primary", elem_classes=["primary"])
                    export_report_btn   = gr.Button(" Export Report (.txt)",
                                                    elem_classes=["warn-btn"])
                    report_file         = gr.File(label="", visible=False)

                with gr.Column(scale=2):
                    circuit_analysis_out = gr.Textbox(
                        label="Circuit Analysis", interactive=False,
                        lines=10, elem_classes=["status-out"]
                    )

            gr.Markdown("##### Issues found", elem_classes=["sec-label"])
            issues_table = gr.Dataframe(
                headers=["Component", "Severity", "Issue", "Suggestion"],
                datatype=["str","str","str","str"],
                interactive=False, wrap=True,
            )

            gr.Markdown("##### Component alternatives  *(click datasheet/buy links)*",
                        elem_classes=["sec-label"])
            suggest_status = gr.Textbox(label="Status", interactive=False,
                                        lines=1, elem_classes=["status-out"])
            alternatives_table = gr.Dataframe(
                headers=["Comp", "Current", "Part Number", "Manufacturer",
                         "Why Better", "Price", "Datasheet", "Buy"],
                datatype=["str","str","str","str","str","str","str","str"],
                interactive=False, wrap=True,
            )
        with gr.Tab("📚Dataset"):
            gr.Markdown(
                "Все сохранённые аннотации. Используется для fine-tuning Qwen2.5-VL.\n\n"
            )
            with gr.Row():
                refresh_btn   = gr.Button("🔄 Refresh",            elem_classes=["secondary"])
                export_ds_btn = gr.Button("📦 Export JSONL",       elem_classes=["warn-btn"])
                ds_stats      = gr.Textbox(label="", interactive=False, scale=3)

            dataset_table = gr.Dataframe(
                headers=["Filename", "Circuit Type", "Components", "Verified", "Date"],
                interactive=False,
            )
            export_ds_file = gr.File(label="Dataset JSONL", visible=False)
    # Tab 1
    analyze_btn.click(on_analyze, [img_input], [ann_out, comp_table, status_out])
    comp_table.change(on_table_change, [comp_table], [ann_out])
    add_btn.click(on_add_component, [new_id, new_type, new_val, comp_table],
                  [comp_table, add_status]).then(on_table_change, [comp_table], [ann_out])
    del_btn.click(on_delete_component, [del_ids, comp_table],
                  [comp_table]).then(on_table_change, [comp_table], [ann_out])
    save_train_btn.click(on_save_for_training, [comp_table], [train_status])
    export_json_btn.click(on_export_json, [comp_table], [export_json_file]).then(
        lambda: gr.update(visible=True), outputs=[export_json_file])

    # Tab 2
    sim_btn.click(on_simulate, [comp_table, sim_type, vcc_slider],
                  [sim_status, netlist_out, meas_out])
    dl_netlist_btn.click(on_download_netlist, [comp_table, sim_type, vcc_slider],
                         [netlist_dl_file]).then(
        lambda: gr.update(visible=True), outputs=[netlist_dl_file])

    # Tab 3
    analyze_circuit_btn.click(on_analyze_circuit, [comp_table],
                               [circuit_analysis_out, issues_table])
    suggest_btn.click(on_get_suggestions, [comp_table, ctx_vcc, ctx_imax, ctx_temp],
                      [suggest_status, alternatives_table])
    export_report_btn.click(on_export_report, [comp_table], [report_file]).then(
        lambda: gr.update(visible=True), outputs=[report_file])

    # Tab 4
    refresh_btn.click(refresh_dataset, outputs=[dataset_table, ds_stats])
    export_ds_btn.click(export_dataset_jsonl, outputs=[export_ds_file]).then(
        lambda: gr.update(visible=True), outputs=[export_ds_file])
    app.load(refresh_dataset, outputs=[dataset_table, ds_stats])


if __name__ == "__main__":
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True,
        theme=gr.themes.Base(),
        css=CSS,
    )
