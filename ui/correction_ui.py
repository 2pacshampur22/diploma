# agent/ui/correction_ui.py
import gradio as gr
import numpy as np
from dataclasses import dataclass, asdict
from typing import List

@dataclass
class Component:
    id: str
    type: str
    value: str
    bbox: List[int]
    confidence: float
    verified: bool = False

COMPONENT_TYPES = [
    "resistor", "capacitor", "inductor", "transistor_npn", 
    "transistor_pnp", "mosfet_n", "mosfet_p", "diode", 
    "zener_diode", "led", "ic", "transformer", "connector",
    "voltage_source", "current_source", "ground", "power"
]

class CorrectionUI:
    def __init__(self, analyzer):
        self.analyzer = analyzer
        self.components = []
        self.image = None
        
    def build_interface(self):
        with gr.Blocks(title="Schematic Analyzer") as app:
            gr.Markdown("## 🔌 AI Schematic Component Analyzer")
            
            with gr.Row():
                with gr.Column(scale=2):
                    image_input = gr.Image(
                        label="Upload Schematic", 
                        type="filepath",
                        tool="boxes"  # Встроенный bbox редактор
                    )
                    analyze_btn = gr.Button("🔍 Analyze", variant="primary")
                
                with gr.Column(scale=3):
                    # Аннотированное изображение
                    annotated_output = gr.AnnotatedImage(
                        label="Detected Components"
                    )
            
            with gr.Row():
                # Таблица компонентов с возможностью редактирования
                components_table = gr.Dataframe(
                    headers=["ID", "Type", "Value", "Confidence", "Verified"],
                    datatype=["str", "str", "str", "number", "bool"],
                    interactive=True,
                    label="Components (editable)"
                )
            
            with gr.Row():
                with gr.Column():
                    # Добавление компонента вручную
                    gr.Markdown("### ➕ Add Component Manually")
                    new_id = gr.Textbox(label="Component ID", placeholder="R5")
                    new_type = gr.Dropdown(COMPONENT_TYPES, label="Type")
                    new_value = gr.Textbox(label="Value", placeholder="10kΩ")
                    add_btn = gr.Button("Add Component")
                
                with gr.Column():
                    gr.Markdown("### 🔄 Actions")
                    export_btn = gr.Button("📤 Export to SPICE Netlist")
                    simulate_btn = gr.Button("⚡ Simulate Circuit")
                    netlist_output = gr.Code(label="SPICE Netlist", language="text")
            
            # Логика событий
            analyze_btn.click(
                fn=self.analyze_schematic,
                inputs=[image_input],
                outputs=[annotated_output, components_table]
            )
            
            export_btn.click(
                fn=self.export_netlist,
                inputs=[components_table],
                outputs=[netlist_output]
            )
            
        return app
    
    def analyze_schematic(self, image_path):
        result = self.analyzer.detect_components(image_path)
        self.components = result.get("components", [])
        
        # Подготовка аннотаций для gr.AnnotatedImage
        img = Image.open(image_path)
        annotations = []
        for comp in self.components:
            bbox = tuple(comp["bbox"])
            label = f"{comp['id']}: {comp['type']}"
            annotations.append((bbox, label))
        
        # Подготовка данных для таблицы
        table_data = [
            [c["id"], c["type"], c.get("value","?"), 
             round(c["confidence"],2), False]
            for c in self.components
        ]
        
        return (img, annotations), table_data