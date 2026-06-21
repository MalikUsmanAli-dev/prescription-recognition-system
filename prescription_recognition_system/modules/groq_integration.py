"""
groq_integration.py
----------------------
Wraps Groq's hosted Llama-4 Vision models (via the official `groq` SDK,
OpenAI-compatible chat.completions endpoint) to extract structured medicine
information from a prescription image.

The model is instructed to return STRICT JSON (enforced via
`response_format={"type": "json_object"}`) describing each medicine found,
its strength, frequency/usage instructions, a per-item confidence score,
and an overall prescription quality assessment.
"""

from __future__ import annotations

import base64
import json
import time
from dataclasses import dataclass, field
from typing import List, Optional

from PIL import Image

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:  # library not installed yet
    GROQ_AVAILABLE = False

from modules.image_processing import pil_image_to_bytes


# Default model - Llama 4 Scout: fast, low-cost, strong vision + JSON support.
# Swap to "meta-llama/llama-4-maverick-17b-128e-instruct" in app Settings
# if you need higher accuracy on messy handwriting.
DEFAULT_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

# Groq's base64 image payloads are capped (~4MB encoded); keep the image
# comfortably under that so large phone-camera photos don't get rejected.
MAX_IMAGE_DIMENSION = 1600
JPEG_QUALITY = 85

SYSTEM_PROMPT = """You are a clinical pharmacy assistant AI specialized in reading
handwritten and printed medical prescriptions, including those common in Pakistan
and South Asia. Carefully analyze the prescription image provided.

Extract every medicine you can identify. For each medicine return:
- name: the medicine / drug name (brand or generic, best guess if partially legible)
- strength: dosage strength if visible (e.g. "500mg"), else null
- frequency: usage instructions / dosing frequency if visible (e.g. "1-0-1 after meals"), else null
- confidence: your confidence (integer 0-100) that this specific reading is correct

Also return:
- overall_confidence: integer 0-100, your overall confidence across the whole prescription
- prescription_quality: one of "Excellent", "Good", "Fair", "Poor" based on legibility/image quality
- doctor_notes: any other readable notes (patient name, diagnosis, follow-up date), else null

Respond with ONLY valid JSON matching this exact schema, no markdown fences, no commentary:
{
  "medicines": [
    {"name": "string", "strength": "string|null", "frequency": "string|null", "confidence": 0}
  ],
  "overall_confidence": 0,
  "prescription_quality": "string",
  "doctor_notes": "string|null"
}

If the image is not a prescription or no medicines are legible, return an empty
"medicines" array and explain why in "doctor_notes".
"""


@dataclass
class MedicineItem:
    name: str
    strength: Optional[str] = None
    frequency: Optional[str] = None
    confidence: int = 0


@dataclass
class AnalysisResult:
    success: bool
    medicines: List[MedicineItem] = field(default_factory=list)
    overall_confidence: int = 0
    prescription_quality: str = "Unknown"
    doctor_notes: Optional[str] = None
    raw_response: Optional[str] = None
    error_message: Optional[str] = None
    processing_time_s: float = 0.0


class GroqPrescriptionAnalyzer:
    """Thin, defensive wrapper around the Groq Vision (Llama-4) chat completions API."""

    def __init__(self, api_key: str, model: str = DEFAULT_MODEL):
        if not GROQ_AVAILABLE:
            raise ImportError(
                "The 'groq' package is not installed. Run: pip install groq"
            )
        if not api_key:
            raise ValueError("A Groq API key is required.")
        self.client = Groq(api_key=api_key)
        self.model = model

    def analyze(self, image: Image.Image) -> AnalysisResult:
        """Send a prescription image to Groq's vision model and parse the structured result."""
        start = time.perf_counter()
        try:
            resized = self._fit_for_upload(image)
            image_bytes = pil_image_to_bytes(resized, fmt="JPEG")
            b64_image = base64.b64encode(image_bytes).decode("utf-8")
            data_url = f"data:image/jpeg;base64,{b64_image}"

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": SYSTEM_PROMPT},
                            {"type": "image_url", "image_url": {"url": data_url}},
                        ],
                    }
                ],
                temperature=0.2,
                response_format={"type": "json_object"},
            )

            elapsed = time.perf_counter() - start
            raw_text = (response.choices[0].message.content or "").strip()
            return self._parse_response(raw_text, elapsed)

        except Exception as exc:  # noqa: BLE001 - surface any API/network error to the UI
            elapsed = time.perf_counter() - start
            return AnalysisResult(
                success=False,
                error_message=self._friendly_error(exc),
                processing_time_s=elapsed,
            )

    # ------------------------------------------------------------------ #
    @staticmethod
    def _fit_for_upload(image: Image.Image) -> Image.Image:
        """Downscale large images so the base64 payload stays under Groq's size limit."""
        w, h = image.size
        longest = max(w, h)
        if longest <= MAX_IMAGE_DIMENSION:
            return image
        scale = MAX_IMAGE_DIMENSION / float(longest)
        new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
        return image.resize(new_size, Image.LANCZOS)

    @staticmethod
    def _parse_response(raw_text: str, elapsed: float) -> AnalysisResult:
        cleaned = raw_text.strip()
        # Defensive cleanup in case the model wraps JSON in markdown fences anyway
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            cleaned = cleaned.replace("json\n", "", 1).replace("json\r\n", "", 1)

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            return AnalysisResult(
                success=False,
                error_message=f"Could not parse AI response as JSON ({exc}).",
                raw_response=raw_text,
                processing_time_s=elapsed,
            )

        medicines = [
            MedicineItem(
                name=item.get("name", "Unknown"),
                strength=item.get("strength"),
                frequency=item.get("frequency"),
                confidence=int(item.get("confidence", 0) or 0),
            )
            for item in data.get("medicines", [])
        ]

        return AnalysisResult(
            success=True,
            medicines=medicines,
            overall_confidence=int(data.get("overall_confidence", 0) or 0),
            prescription_quality=data.get("prescription_quality", "Unknown"),
            doctor_notes=data.get("doctor_notes"),
            raw_response=raw_text,
            processing_time_s=elapsed,
        )

    @staticmethod
    def _friendly_error(exc: Exception) -> str:
        message = str(exc)
        lowered = message.lower()
        if "api key" in lowered or "permission" in lowered or "unauthenticated" in lowered or "401" in lowered:
            return "Invalid or missing Groq API key. Please check your key in Settings."
        if "quota" in lowered or "rate" in lowered or "429" in lowered:
            return "Groq API rate limit / quota exceeded. Please wait a moment and try again."
        if "timeout" in lowered or "deadline" in lowered:
            return "The request to Groq timed out. Please check your connection and retry."
        if "decommission" in lowered or "not found" in lowered or "404" in lowered:
            return f"Model '{message}' is unavailable. Try a different model in Settings."
        return f"Groq API request failed: {message}"
