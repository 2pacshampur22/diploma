from __future__ import annotations
import base64
import json
import re
from pathlib import Path
from PIL import Image

# Определяем доступные бэкенды

# Бэкенд 1: qwen-agent (pip install qwen-agent)
QWEN_AGENT_AVAILABLE = False
try:
    from qwen_agent.llm import get_chat_model
    QWEN_AGENT_AVAILABLE = True
except ImportError:
    pass

# Бэкенд 2: transformers + qwen-vl-utils
TRANSFORMERS_AVAILABLE = False
QWEN_VL_UTILS_AVAILABLE = False
QwenModelClass = None

try:
    from transformers import AutoProcessor
    TRANSFORMERS_AVAILABLE = True

    # Пробуем новый класс (transformers >= 4.52)
    try:
        from transformers import Qwen2_5_VLForConditionalGeneration
        QwenModelClass = Qwen2_5_VLForConditionalGeneration
    except ImportError:
        pass

    # Старый класс (transformers 4.45–4.51)
    if QwenModelClass is None:
        try:
            from transformers import Qwen2VLForConditionalGeneration
            QwenModelClass = Qwen2VLForConditionalGeneration
        except ImportError:
            pass
except ImportError:
    pass

try:
    from qwen_vl_utils import process_vision_info
    QWEN_VL_UTILS_AVAILABLE = True
except ImportError:
    pass

TRANSFORMERS_BACKEND_AVAILABLE = (QwenModelClass is not None and QWEN_VL_UTILS_AVAILABLE)

# Итоговый статус
_backends_status = {
    "qwen_agent":   QWEN_AGENT_AVAILABLE,
    "transformers": TRANSFORMERS_BACKEND_AVAILABLE,
    "mock":         True,
}

_best_backend = next(k for k, v in _backends_status.items() if v)

print(f" SchematicAnalyzer backends: "
      + " | ".join(f"{k}={'V' if v else 'X'}" for k, v in _backends_status.items()))
print(f"   Selected: {_best_backend}")

if _best_backend == "mock":
    print("   → Install qwen-agent:  pip install qwen-agent")
    print("   → Or transformers:     pip install transformers qwen-vl-utils accelerate torch")


# Промпты

SYSTEM_PROMPT = """You are an expert electronics engineer specializing in schematic analysis.
Identify ALL electronic components visible in the circuit schematic image.

For each component return:
- id: unique reference designator (R1, C2, Q1, U1, etc.)
- type: one of: resistor / capacitor / inductor / transistor_npn / transistor_pnp /
        mosfet_n / mosfet_p / diode / zener_diode / led / ic / transformer /
        connector / voltage_source / current_source / ground / power / unknown
- value: component value or model if visible (e.g. "10kΩ", "100nF", "BC547", "LM358")
- bbox: bounding box [x1, y1, x2, y2] in pixels (top-left to bottom-right)
- confidence: float 0.0–1.0

Also identify:
- circuit_type: amplifier / filter / power_supply / oscillator / digital / rf / mixed / other

Return ONLY valid JSON, no markdown fences, no explanation:
{
  "components": [
    {"id": "R1", "type": "resistor", "value": "10kΩ", "bbox": [x1,y1,x2,y2], "confidence": 0.95}
  ],
  "circuit_type": "amplifier",
  "warnings": []
}"""

USER_PROMPT = ("Analyze this electrical schematic image. "
               "Find and locate all electronic components. "
               "Return JSON only.")



def _image_to_base64(image_path: str) -> str:
    with open(image_path, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")
    ext = Path(image_path).suffix.lower().lstrip(".")
    mime = {"jpg": "jpeg", "jpeg": "jpeg", "png": "png",
            "bmp": "bmp", "webp": "webp"}.get(ext, "png")
    return f"data:image/{mime};base64,{data}"


# Mock бэкенд

class _MockBackend:
    def run(self, image_path: str) -> dict:
        img  = Image.open(image_path)
        w, h = img.size
        return {
            "components": [
                {"id": "R1", "type": "resistor",       "value": "10kΩ",  "bbox": [int(w*.08), int(h*.10), int(w*.20), int(h*.22)], "confidence": 0.95},
                {"id": "R2", "type": "resistor",       "value": "4.7kΩ", "bbox": [int(w*.28), int(h*.10), int(w*.42), int(h*.22)], "confidence": 0.91},
                {"id": "C1", "type": "capacitor",      "value": "100nF", "bbox": [int(w*.50), int(h*.10), int(w*.63), int(h*.22)], "confidence": 0.88},
                {"id": "Q1", "type": "transistor_npn", "value": "BC547", "bbox": [int(w*.68), int(h*.10), int(w*.85), int(h*.35)], "confidence": 0.93},
                {"id": "D1", "type": "diode",          "value": "1N4148","bbox": [int(w*.08), int(h*.50), int(w*.25), int(h*.62)], "confidence": 0.97},
                {"id": "L1", "type": "inductor",       "value": "100µH", "bbox": [int(w*.38), int(h*.50), int(w*.55), int(h*.62)], "confidence": 0.82},
            ],
            "circuit_type": "amplifier",
            "warnings": ["Mock mode — install qwen-agent or transformers for real inference"],
        }


# qwen-agent бэкенд

class _QwenAgentBackend:
    def __init__(self, model_name: str, device: str = "auto"):
        self.model_name = model_name

        # Определяем конфиг в зависимости от того локальная модель или API
        if model_name.startswith("Qwen/") or "/" not in model_name or Path(model_name).exists():
            # Локальная модель через transformers
            cfg = {
                "model_type":    "qwenvl_transformers",
                "model_id_or_path": model_name,
                "device":        device,
                "generate_cfg": {
                    "max_new_tokens": 512,
                    "do_sample":      False,
                },
            }
        else:
            # OpenAI-совместимый API (например vLLM или DashScope)
            cfg = {
                "model":    model_name,
                "api_base": "https://dashscope.aliyuncs.com/compatible-mode/v1",
                "api_key":  "EMPTY",
            }

        print(f"qwen-agent: loading {model_name}")
        self._llm = get_chat_model(cfg)
        print("qwen-agent: model ready")

    def run(self, image_path: str) -> str:
        """Вызывает модель и возвращает сырой текст ответа."""
        img_b64 = _image_to_base64(image_path)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    # qwen-agent принимает изображение как {"image": "data:..."}
                    {"image": img_b64},
                    {"text":  USER_PROMPT},
                ],
            },
        ]

        responses = []
        for chunk in self._llm.chat(messages=messages, stream=True):
            responses.append(chunk)

        if responses:
            last = responses[-1]
            if isinstance(last, list) and last:
                return last[-1].get("content", "")
            if isinstance(last, dict):
                return last.get("content", "")
        return ""


# transformers бэкенд

class _TransformersBackend:
    def __init__(self, model_name: str, device: str = "auto"):
        print(f"transformers: loading {model_name}")
        self._model = QwenModelClass.from_pretrained(
            model_name, torch_dtype="auto", device_map=device
        )
        self._processor = AutoProcessor.from_pretrained(model_name)
        print("transformers: model ready")

    def run(self, image_path: str) -> str:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text",  "text":  USER_PROMPT},
                ],
            },
        ]
        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self._processor(
            text=[text], images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt",
        ).to(self._model.device)

        ids = self._model.generate(
            **inputs, max_new_tokens=512, do_sample=False
        )
        trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, ids)]
        return self._processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]


# Основной класс

class SchematicAnalyzer:
    """
    Анализатор электрических схем с автоматическим выбором бэкенда.

    Порядок приоритета: qwen-agent - transformers - mock

    Args:
        model_name: HuggingFace ID или путь к локальной модели.
                    По умолчанию "Qwen/Qwen2.5-VL-7B-Instruct".
        device:     "auto" | "cuda" | "cpu" | "mps"
        backend:    Принудительно выбрать бэкенд: "qwen_agent" | "transformers" | "mock"
        min_confidence: Порог уверенности (0.0–1.0), компоненты ниже порога отбрасываются.
    """

    def __init__(
        self,
        model_name:     str   = "Qwen/Qwen2.5-VL-7B-Instruct",
        device:         str   = "auto",
        backend:        str   = "auto",
        min_confidence: float = 0.0,
    ):
        self.min_confidence = min_confidence
        self._raw_backend   = None
        self.backend_name   = "mock"

        # Принудительный выбор бэкенда
        if backend == "mock":
            self._raw_backend = _MockBackend()
            self.backend_name = "mock"
            print("SchematicAnalyzer: forced mock backend")
            return

        # Автоматический выбор
        chosen = backend if backend != "auto" else _best_backend

        if chosen == "qwen_agent" and QWEN_AGENT_AVAILABLE:
            try:
                self._raw_backend = _QwenAgentBackend(model_name, device)
                self.backend_name = "qwen_agent"
                return
            except Exception as e:
                print(f"qwen-agent failed ({e}), trying next backend...")

        if chosen in ("transformers", "qwen_agent") and TRANSFORMERS_BACKEND_AVAILABLE:
            try:
                self._raw_backend = _TransformersBackend(model_name, device)
                self.backend_name = "transformers"
                return
            except Exception as e:
                print(f"transformers failed ({e}), falling back to mock...")

        # Fallback
        print("  SchematicAnalyzer: using mock backend")
        self._raw_backend = _MockBackend()
        self.backend_name = "mock"

    # ── Публичный API ─────────────────────────────────────────────────────────

    def detect_components(self, image_path: str) -> dict:
        """
        Анализирует схему и возвращает компоненты.

        Returns:
            {
                "components":   [{"id", "type", "value", "bbox", "confidence"}, ...],
                "circuit_type": "amplifier" | "filter" | ...,
                "warnings":     [...],
            }
        """
        if isinstance(self._raw_backend, _MockBackend):
            return self._raw_backend.run(image_path)

        # Реальный инференс
        image    = Image.open(image_path).convert("RGB")
        raw_text = self._raw_backend.run(image_path)
        return self._parse_and_validate(raw_text, image)

    @classmethod
    def from_finetuned(cls, model_path: str, **kwargs) -> "SchematicAnalyzer":
        """Загружает дообученную модель по локальному пути."""
        return cls(model_name=model_path, **kwargs)

    # ── Приватные ─────────────────────────────────────────────────────────────

    def _parse_and_validate(self, response: str, image: Image.Image) -> dict:
        """Парсит JSON и валидирует bbox под размеры изображения."""
        # Убираем markdown обёртку ```json ... ```
        response = re.sub(r"```(?:json)?\s*", "", response).strip().rstrip("`").strip()

        # Пробуем распарсить напрямую
        data = None
        try:
            data = json.loads(response)
        except json.JSONDecodeError:
            # Ищем первый JSON объект в тексте
            m = re.search(r'\{.*\}', response, re.DOTALL)
            if m:
                try:
                    data = json.loads(m.group())
                except json.JSONDecodeError:
                    pass

        if data is None:
            return {
                "components":   [],
                "circuit_type": "unknown",
                "warnings":     [f"Failed to parse model response: {response[:300]}"],
            }

        # Валидация и клиппинг bbox
        w, h       = image.size
        warnings   = list(data.get("warnings", []))
        valid      = []

        for comp in data.get("components", []):
            bbox = comp.get("bbox", [])
            if len(bbox) == 4:
                x1, y1, x2, y2 = bbox
                comp["bbox"] = [
                    max(0, min(int(x1), w)), max(0, min(int(y1), h)),
                    max(0, min(int(x2), w)), max(0, min(int(y2), h)),
                ]
            else:
                warnings.append(f"Bad bbox for {comp.get('id','?')}: {bbox}")
                comp["bbox"] = []

            if comp.get("confidence", 1.0) >= self.min_confidence:
                valid.append(comp)

        return {
            "components":   valid,
            "circuit_type": data.get("circuit_type", "unknown"),
            "warnings":     warnings,
        }