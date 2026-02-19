"""
Fine-tuning pipeline для Qwen2.5-VL на задаче детекции компонентов схем.

Использует LLaMA-Factory или прямой Hugging Face Trainer.
"""

import json
import os
import random
import base64
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field, asdict
from datetime import datetime
import re


# ─── Конфигурация ─────────────────────────────────────────────────────────────
@dataclass
class FinetuneConfig:
    # Модель
    model_name:          str  = "Qwen/Qwen2.5-VL-7B-Instruct"
    output_dir:          str  = "models/qwen-schematic-v1"
    
    # Данные
    annotated_dir:       str  = "data/annotated"
    processed_dir:       str  = "data/processed"
    train_split:         float = 0.85
    val_split:           float = 0.10
    test_split:          float = 0.05
    
    # Обучение
    num_epochs:          int   = 3
    batch_size:          int   = 1          # VL модели требуют много памяти
    gradient_accumulation: int = 8
    learning_rate:       float = 2e-5
    max_seq_len:         int   = 4096
    lora_rank:           int   = 16         # LoRA для экономии памяти
    lora_alpha:          int   = 32
    lora_dropout:        float = 0.05
    
    # Железо
    bf16:                bool  = True
    use_flash_attention: bool  = True
    gradient_checkpointing: bool = True
    
    # Логирование
    logging_steps:       int   = 10
    eval_steps:          int   = 100
    save_steps:          int   = 200
    wandb_project:       str   = "qwen-schematic-agent"


# ─── Системный промпт ─────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are an expert electronics engineer and schematic analyzer.
Your task is to identify and locate all electronic components in circuit schematics.

For each component you MUST provide:
- Unique ID (e.g. R1, C2, Q1)
- Component type (resistor/capacitor/inductor/transistor_npn/transistor_pnp/
  mosfet_n/mosfet_p/diode/zener_diode/led/ic/transformer/connector/
  voltage_source/current_source/ground/power)
- Value if visible (e.g. 10kΩ, 100nF, 2N2222)
- Bounding box as [x1, y1, x2, y2] in pixels
- Confidence score 0.0–1.0

Return ONLY a valid JSON object. No markdown, no explanation.
Schema:
{
  "components": [
    {
      "id": "string",
      "type": "string",
      "value": "string or null",
      "bbox": [x1, y1, x2, y2],
      "confidence": float
    }
  ],
  "circuit_type": "string (amplifier/filter/power_supply/oscillator/digital/other)",
  "warnings": ["string"]
}"""

USER_PROMPT = "Analyze this electrical schematic. Identify all components with their bounding boxes and values."


# ─── Подготовка данных ────────────────────────────────────────────────────────
class DatasetBuilder:
    """
    Строит датасет для Qwen2.5-VL fine-tuning из аннотированных схем.
    
    Поддерживаемые форматы выхода:
    - LLaMA-Factory JSON
    - Hugging Face datasets (Arrow)
    - JSONL (universal)
    """
    
    def __init__(self, config: FinetuneConfig):
        self.cfg = config
        os.makedirs(config.processed_dir, exist_ok=True)
    
    def image_to_base64(self, image_path: str) -> str:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    
    def annotation_to_conversation(self, annotation: dict, image_path: str) -> dict:
        """
        Конвертирует одну аннотацию в conversation-формат для Qwen-VL.
        
        Формат LLaMA-Factory multimodal:
        {
          "messages": [
            {"role": "system",    "content": "..."},
            {"role": "user",      "content": [{"type": "image"}, {"type": "text", "text": "..."}]},
            {"role": "assistant", "content": "...json..."}
          ],
          "images": ["path/to/image.png"]
        }
        """
        # Формируем ответ ассистента
        components = annotation.get("components", [])
        circuit_type = annotation.get("circuit_type", "unknown")
        
        answer = json.dumps({
            "components": [
                {
                    "id":         c["id"],
                    "type":       c["type"],
                    "value":      c.get("value"),
                    "bbox":       c.get("bbox", []),
                    "confidence": round(c.get("confidence", 1.0), 2)
                }
                for c in components if c.get("verified", True)
            ],
            "circuit_type": circuit_type,
            "warnings":     []
        }, ensure_ascii=False, separators=(",", ":"))
        
        return {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": USER_PROMPT}
                    ]
                },
                {"role": "assistant", "content": answer}
            ],
            "images": [image_path]
        }
    
    def build_augmented_variants(self, annotation: dict, image_path: str) -> list:
        """
        Создаёт несколько вариантов промпта для одной схемы (data augmentation на уровне текста).
        """
        variants = []
        
        # Вариант 1: базовый
        variants.append(self.annotation_to_conversation(annotation, image_path))
        
        # Вариант 2: спросить только определённый тип
        resistors = [c for c in annotation.get("components", []) if c["type"] == "resistor"]
        if resistors:
            answer = json.dumps({
                "components": resistors,
                "circuit_type": annotation.get("circuit_type", "unknown"),
                "warnings": []
            }, ensure_ascii=False, separators=(",", ":"))
            variants.append({
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": "List only the RESISTORS in this schematic with their values and locations."}
                        ]
                    },
                    {"role": "assistant", "content": answer}
                ],
                "images": [image_path]
            })
        
        # Вариант 3: вопрос о типе схемы
        variants.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": f"What type of circuit is this? Return JSON with circuit_type and a brief description."}
                    ]
                },
                {
                    "role": "assistant",
                    "content": json.dumps({
                        "circuit_type": annotation.get("circuit_type", "unknown"),
                        "description": f"This is a {annotation.get('circuit_type','unknown')} circuit with {len(annotation.get('components',[]))} components."
                    })
                }
            ],
            "images": [image_path]
        })
        
        return variants
    
    def load_annotations(self) -> list:
        ann_dir = Path(self.cfg.annotated_dir)
        records = []
        
        for json_file in ann_dir.glob("*.json"):
            try:
                with open(json_file) as f:
                    annotation = json.load(f)
                
                # Ищем соответствующее изображение
                stem = json_file.stem
                image_path = None
                for ext in [".png", ".jpg", ".jpeg", ".bmp", ".tif"]:
                    candidate = ann_dir / (stem + ext)
                    if candidate.exists():
                        image_path = str(candidate)
                        break
                
                if image_path:
                    records.append((annotation, image_path))
                else:
                    print(f"⚠️  No image found for {json_file.name}")
            
            except Exception as e:
                print(f"⚠️  Error loading {json_file}: {e}")
        
        print(f"📚 Loaded {len(records)} annotated schematics")
        return records
    
    def build_and_split(self, augment: bool = True) -> dict:
        """
        Строит финальный датасет и разбивает на train/val/test.
        
        Returns: {"train": [...], "val": [...], "test": [...]}
        """
        records = self.load_annotations()
        
        if not records:
            print("❌ No annotations found. Add some via the UI first!")
            return {"train": [], "val": [], "test": []}
        
        # Генерируем примеры
        all_examples = []
        for annotation, image_path in records:
            if augment:
                examples = self.build_augmented_variants(annotation, image_path)
            else:
                examples = [self.annotation_to_conversation(annotation, image_path)]
            all_examples.extend(examples)
        
        # Перемешиваем
        random.seed(42)
        random.shuffle(all_examples)
        
        n = len(all_examples)
        n_train = int(n * self.cfg.train_split)
        n_val   = int(n * self.cfg.val_split)
        
        splits = {
            "train": all_examples[:n_train],
            "val":   all_examples[n_train:n_train+n_val],
            "test":  all_examples[n_train+n_val:],
        }
        
        # Сохраняем
        for split_name, data in splits.items():
            # JSONL формат
            jsonl_path = f"{self.cfg.processed_dir}/{split_name}.jsonl"
            with open(jsonl_path, "w", encoding="utf-8") as f:
                for item in data:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
            
            # LLaMA-Factory JSON формат
            lf_path = f"{self.cfg.processed_dir}/{split_name}_llamafactory.json"
            with open(lf_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            print(f"✅ {split_name:5s}: {len(data):4d} examples → {jsonl_path}")
        
        # Статистика
        self._print_stats(splits)
        
        return splits
    
    def _print_stats(self, splits: dict):
        print("\n📊 Dataset Statistics:")
        print("─" * 50)
        total = sum(len(v) for v in splits.values())
        for split, data in splits.items():
            print(f"  {split:8s}: {len(data):4d} ({len(data)/total*100:.1f}%)")
        
        # Считаем типы компонентов
        type_counts = {}
        for data in splits.values():
            for example in data:
                try:
                    answer = example["messages"][-1]["content"]
                    parsed = json.loads(answer)
                    for comp in parsed.get("components", []):
                        t = comp.get("type", "unknown")
                        type_counts[t] = type_counts.get(t, 0) + 1
                except Exception:
                    pass
        
        if type_counts:
            print("\n  Component distribution:")
            for ctype, count in sorted(type_counts.items(), key=lambda x: -x[1]):
                print(f"    {ctype:20s}: {count}")


# ─── LLaMA-Factory конфиг ─────────────────────────────────────────────────────
def generate_llamafactory_config(cfg: FinetuneConfig) -> dict:
    """
    Генерирует YAML конфиг для LLaMA-Factory.
    Запуск: llamafactory-cli train config/lf_train.yaml
    """
    return {
        "### model": None,
        "model_name_or_path": cfg.model_name,
        
        "### method": None,
        "stage":           "sft",
        "do_train":        True,
        "finetuning_type": "lora",
        "lora_rank":       cfg.lora_rank,
        "lora_alpha":      cfg.lora_alpha,
        "lora_dropout":    cfg.lora_dropout,
        "lora_target":     "all",
        
        "### dataset": None,
        "dataset":          "schematic_train",
        "dataset_dir":      cfg.processed_dir,
        "template":         "qwen2_vl",
        "cutoff_len":       cfg.max_seq_len,
        "overwrite_cache":  True,
        
        "### output": None,
        "output_dir":          cfg.output_dir,
        "logging_steps":       cfg.logging_steps,
        "save_steps":          cfg.save_steps,
        "plot_loss":           True,
        "overwrite_output_dir": True,
        
        "### train": None,
        "per_device_train_batch_size": cfg.batch_size,
        "gradient_accumulation_steps": cfg.gradient_accumulation,
        "learning_rate":               cfg.learning_rate,
        "num_train_epochs":            cfg.num_epochs,
        "lr_scheduler_type":           "cosine",
        "warmup_ratio":                0.1,
        "bf16":                        cfg.bf16,
        "flash_attn":                  "fa2" if cfg.use_flash_attention else "disabled",
        
        "### eval": None,
        "val_size":                    cfg.val_split,
        "per_device_eval_batch_size":  cfg.batch_size,
        "eval_strategy":               "steps",
        "eval_steps":                  cfg.eval_steps,
    }


def generate_llamafactory_dataset_info(cfg: FinetuneConfig) -> dict:
    """
    dataset_info.json для LLaMA-Factory
    """
    return {
        "schematic_train": {
            "file_name":   "train_llamafactory.json",
            "formatting":  "sharegpt",
            "columns": {
                "messages": "messages",
                "images":   "images"
            },
        },
        "schematic_val": {
            "file_name":   "val_llamafactory.json",
            "formatting":  "sharegpt",
            "columns": {
                "messages": "messages",
                "images":   "images"
            },
        }
    }


# ─── Скрипт запуска обучения ──────────────────────────────────────────────────
TRAIN_SCRIPT = '''#!/bin/bash
# ─── Fine-tuning Qwen2.5-VL для детекции компонентов схем ───────────────────
# Требования: GPU 24GB+ (RTX 3090/4090 или A100)

set -e

echo "🔧 Installing dependencies..."
pip install -q llamafactory transformers accelerate peft datasets
pip install -q qwen-vl-utils pillow opencv-python

echo "📦 Preparing dataset..."
python finetuning/prepare_dataset.py

echo "🚀 Starting training..."
llamafactory-cli train config/lf_train.yaml

echo "✅ Training complete! Model saved to: models/qwen-schematic-v1"

# Опционально: слияние LoRA весов
echo "🔗 Merging LoRA weights..."
llamafactory-cli export config/lf_export.yaml
'''

DOCKER_COMPOSE = '''version: "3.9"
services:
  trainer:
    image: hiyouga/llamafactory:latest
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
      - WANDB_API_KEY=${WANDB_API_KEY}
    volumes:
      - .:/workspace
      - ~/.cache/huggingface:/root/.cache/huggingface
    working_dir: /workspace
    command: >
      bash -c "pip install -q qwen-vl-utils &&
               llamafactory-cli train config/lf_train.yaml"
    shm_size: "16gb"
    
  ui:
    build: .
    ports:
      - "7860:7860"
    volumes:
      - .:/workspace
    command: python ui/app.py
'''

DOCKERFILE = '''FROM python:3.11-slim

WORKDIR /workspace

RUN apt-get update && apt-get install -y \\
    libgl1-mesa-glx libglib2.0-0 \\
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 7860
CMD ["python", "ui/app.py"]
'''


# ─── Скрипт генерации синтетических данных ────────────────────────────────────
SYNTHETIC_DATA_SCRIPT = '''"""
Генерация синтетических схем для расширения датасета.
Использует schemdraw для рисования схем программно.

pip install schemdraw
"""
import schemdraw
import schemdraw.elements as elm
import json, random, os
from pathlib import Path


def generate_rc_filter(output_dir: str, idx: int):
    """RC фильтр нижних частот"""
    with schemdraw.Drawing(show=False) as d:
        d.config(fontsize=12)
        
        r_val = random.choice(["1kΩ", "4.7kΩ", "10kΩ", "47kΩ", "100kΩ"])
        c_val = random.choice(["10nF", "47nF", "100nF", "470nF", "1µF"])
        
        V  = d.add(elm.SourceV().up().label("Vin"))
        R1 = d.add(elm.Resistor().right().label(r_val))
        C1 = d.add(elm.Capacitor().down().label(c_val))
        d.add(elm.Line().left())
        
        fname = f"{output_dir}/rc_filter_{idx:04d}.png"
        d.save(fname, dpi=150)
        
        annotation = {
            "image": os.path.basename(fname),
            "circuit_type": "filter",
            "components": [
                {"id": "R1", "type": "resistor",  "value": r_val, "bbox": [], "confidence": 1.0, "verified": True},
                {"id": "C1", "type": "capacitor", "value": c_val, "bbox": [], "confidence": 1.0, "verified": True},
                {"id": "V1", "type": "voltage_source", "value": "Vin", "bbox": [], "confidence": 1.0, "verified": True},
            ],
            "synthetic": True
        }
        return annotation, fname


def generate_transistor_amplifier(output_dir: str, idx: int):
    """Усилитель на одном транзисторе"""
    with schemdraw.Drawing(show=False) as d:
        d.config(fontsize=12)
        
        rc_val = random.choice(["1kΩ", "2.2kΩ", "4.7kΩ"])
        rb_val = random.choice(["47kΩ", "100kΩ", "220kΩ"])
        
        d.add(elm.BjtNpn(circle=True).anchor("base").label("Q1\\nBC547"))
        d.add(elm.Resistor().up().at("collector").label(rc_val))
        d.add(elm.Dot().label("Vcc", loc="right"))
        d.add(elm.Resistor().left().at("base").label(rb_val))
        d.add(elm.Dot().label("Vin", loc="left"))
        d.add(elm.Ground().at("emitter"))
        
        fname = f"{output_dir}/amp_{idx:04d}.png"
        d.save(fname, dpi=150)
        
        annotation = {
            "image": os.path.basename(fname),
            "circuit_type": "amplifier",
            "components": [
                {"id": "Q1",  "type": "transistor_npn", "value": "BC547", "bbox": [], "confidence": 1.0, "verified": True},
                {"id": "RC1", "type": "resistor",        "value": rc_val,  "bbox": [], "confidence": 1.0, "verified": True},
                {"id": "RB1", "type": "resistor",        "value": rb_val,  "bbox": [], "confidence": 1.0, "verified": True},
            ],
            "synthetic": True
        }
        return annotation, fname


if __name__ == "__main__":
    out = "data/annotated"
    os.makedirs(out, exist_ok=True)
    
    count = 0
    for i in range(50):
        ann, _ = generate_rc_filter(out, i)
        json_path = f"{out}/rc_filter_{i:04d}.json"
        with open(json_path, "w") as f:
            json.dump(ann, f, indent=2)
        count += 1
    
    for i in range(30):
        ann, _ = generate_transistor_amplifier(out, i)
        json_path = f"{out}/amp_{i:04d}.json"
        with open(json_path, "w") as f:
            json.dump(ann, f, indent=2)
        count += 1
    
    print(f"✅ Generated {count} synthetic schematics")
'''


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    cfg = FinetuneConfig()
    
    # Создаём директории
    os.makedirs(cfg.processed_dir, exist_ok=True)
    os.makedirs("config", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    
    # Строим датасет
    builder = DatasetBuilder(cfg)
    splits  = builder.build_and_split(augment=True)
    
    # Сохраняем LLaMA-Factory конфиги
    lf_cfg = generate_llamafactory_config(cfg)
    
    # Конвертируем в YAML (без зависимости от pyyaml)
    yaml_lines = []
    for k, v in lf_cfg.items():
        if v is None:
            yaml_lines.append(f"\n{k}")
        elif isinstance(v, bool):
            yaml_lines.append(f"{k}: {'true' if v else 'false'}")
        else:
            yaml_lines.append(f"{k}: {v}")
    
    with open("config/lf_train.yaml", "w") as f:
        f.write("\n".join(yaml_lines))
    print("✅ Saved config/lf_train.yaml")
    
    # dataset_info.json
    ds_info = generate_llamafactory_dataset_info(cfg)
    with open(f"{cfg.processed_dir}/dataset_info.json", "w") as f:
        json.dump(ds_info, f, indent=2)
    print(f"✅ Saved {cfg.processed_dir}/dataset_info.json")
    
    print("\n🚀 Next steps:")
    print("  1. Add annotated schematics via:  python ui/app.py")
    print("  2. Run synthetic data generation: python finetuning/generate_synthetic.py")
    print("  3. Prepare dataset:               python finetuning/prepare_dataset.py")
    print("  4. Start training:                llamafactory-cli train config/lf_train.yaml")
    print("     OR with Docker:                docker-compose up trainer")


if __name__ == "__main__":
    main()
