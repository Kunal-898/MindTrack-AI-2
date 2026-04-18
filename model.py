"""
model.py — DistilRoBERTa Emotion Classifier
Loads your fine-tuned RobertaForSequenceClassification model from ./model_files/

Model config (from your config.json):
  architecture : RobertaForSequenceClassification
  base         : distilroberta-base  (6 hidden layers, hidden_size=768)
  num_labels   : 6
  label map    : 0=anger  1=disgust  2=fear  3=joy  4=neutral  5=sadness

MindTrack emotion mapping:
  anger   → Anger
  disgust → Anger   (merged — no disgust slot in MindTrack UI)
  fear    → Fear
  joy     → Joy
  neutral → Calm
  sadness → Sadness
  (Stress is derived from fear+anger when both are elevated)
"""

import os
import json
import math
import logging
from pathlib import Path

logger = logging.getLogger("mindtrack.model")

# ─── Paths ────────────────────────────────────────────────────────────────────

MODEL_DIR = Path(__file__).parent / "model_files"

# ─── Label mappings ───────────────────────────────────────────────────────────

# Raw model output label  →  MindTrack UI emotion name
# "disgust" is merged into Anger since MindTrack has no disgust slot.
RAW_TO_MINDTRACK = {
    "anger":   "Anger",
    "disgust": "Anger",    # merged
    "fear":    "Fear",
    "joy":     "Joy",
    "neutral": "Calm",
    "sadness": "Sadness",
}

# All 6 MindTrack emotions (Stress is synthetic — derived below)
MINDTRACK_EMOTIONS = ["Joy", "Calm", "Sadness", "Stress", "Anger", "Fear"]

# Wellness contribution weight per emotion (0–1, higher = healthier)
WELLNESS_WEIGHTS = {
    "Joy":     1.00,
    "Calm":    0.85,
    "Sadness": 0.25,
    "Fear":    0.20,
    "Anger":   0.10,
    "Stress":  0.15,
}


class EmotionClassifier:
    """
    Wraps your fine-tuned DistilRoBERTa emotion classifier.

    Loading priority:
      1. ./model_files/   — your uploaded fine-tuned weights  ← preferred
      2. j-hartmann/emotion-english-distilroberta-base        ← HuggingFace Hub fallback

    Place your pytorch_model.bin (or model.safetensors) inside model_files/
    alongside the config.json and tokenizer files already there.
    """

    HUB_FALLBACK = "j-hartmann/emotion-english-distilroberta-base"

    def __init__(self):
        self.pipeline   = None
        self.model_name = "not loaded"
        self.id2label   = {}
        self._load()

    # ── Loading ───────────────────────────────────────────────────────────────

    def _load(self):
        try:
            from transformers import pipeline as hf_pipeline

            has_weights = MODEL_DIR.is_dir() and any(
                (MODEL_DIR / f).exists()
                for f in ["pytorch_model.bin", "model.safetensors"]
            )

            if has_weights:
                logger.info(f"Loading local model from {MODEL_DIR} ...")
                self.pipeline = hf_pipeline(
                    task="text-classification",
                    model=str(MODEL_DIR),
                    tokenizer=str(MODEL_DIR),
                    top_k=None,          # return all label scores
                    truncation=True,
                    max_length=512,
                )
                self.model_name = "local/distilroberta-emotion"
                logger.info("✅ Local DistilRoBERTa model loaded")
            else:
                logger.warning(
                    "No model weights found in ./model_files/. "
                    f"Falling back to HuggingFace Hub: {self.HUB_FALLBACK}"
                )
                self.pipeline = hf_pipeline(
                    task="text-classification",
                    model=self.HUB_FALLBACK,
                    top_k=None,
                    truncation=True,
                    max_length=512,
                )
                self.model_name = self.HUB_FALLBACK
                logger.info(f"✅ Hub model loaded: {self.HUB_FALLBACK}")

            # Read id2label from config for logging / validation
            cfg_path = MODEL_DIR / "config.json"
            if cfg_path.exists():
                with open(cfg_path) as f:
                    cfg = json.load(f)
                self.id2label = cfg.get("id2label", {})
                logger.info(f"Label map: {self.id2label}")

        except ImportError:
            logger.error("transformers not installed — run: pip install transformers torch")
        except Exception as e:
            logger.error(f"Model load error: {e}")

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict(self, text: str) -> dict:
        """
        Run emotion classification on `text`.

        Returns:
        {
            "emotions":      [{"emotion": str, "percentage": float}, ...],
            "dominant":      str,
            "wellnessIndex": int   (0-100)
        }
        """
        if self.pipeline is None:
            logger.warning("Model not loaded — using keyword fallback")
            return self._keyword_fallback(text)

        try:
            raw = self.pipeline(text)

            # HF returns [[{label, score}, ...]] when top_k=None
            if isinstance(raw, list) and isinstance(raw[0], list):
                raw = raw[0]

            # ── Step 1: aggregate raw scores into MindTrack buckets ──────────
            bucket: dict[str, float] = {e: 0.0 for e in MINDTRACK_EMOTIONS}

            for item in raw:
                label = item["label"].lower()
                score = float(item["score"])
                mt_label = RAW_TO_MINDTRACK.get(label)
                if mt_label:
                    bucket[mt_label] += score   # disgust + anger both add to Anger

            # ── Step 2: derive Stress from elevated fear + anger ─────────────
            # If fear+anger together exceed 35% of total raw signal, carve out
            # a Stress component so the UI reflects chronic tension.
            combined_threat = bucket["Fear"] + bucket["Anger"]
            total_raw       = sum(bucket.values()) or 1.0
            threat_ratio    = combined_threat / total_raw

            if threat_ratio > 0.35:
                # Transfer 40% of the threat signal into Stress
                transfer = combined_threat * 0.40
                bucket["Fear"]   *= 0.75
                bucket["Anger"]  *= 0.75
                bucket["Stress"] += transfer

            # ── Step 3: normalise to percentages ────────────────────────────
            total = sum(bucket.values()) or 1.0
            emotions = sorted(
                [
                    {
                        "emotion":    name,
                        "percentage": round((score / total) * 100, 1),
                    }
                    for name, score in bucket.items()
                ],
                key=lambda x: x["percentage"],
                reverse=True,
            )

            # Enforce percentages sum to exactly 100
            total_pct = sum(e["percentage"] for e in emotions)
            if total_pct != 100.0:
                emotions[0]["percentage"] = round(
                    emotions[0]["percentage"] + (100.0 - total_pct), 1
                )

            dominant      = emotions[0]["emotion"]
            wellness_index = self._wellness(bucket, total)

            return {
                "emotions":      emotions,
                "dominant":      dominant,
                "wellnessIndex": wellness_index,
            }

        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return self._keyword_fallback(text)

    # ── Wellness Index ─────────────────────────────────────────────────────────

    def _wellness(self, bucket: dict, total: float) -> int:
        """
        Compute a 0–100 wellness index as a weighted sum of
        normalised emotion probabilities × wellness weight.
        """
        score = sum(
            (bucket.get(em, 0.0) / total) * w
            for em, w in WELLNESS_WEIGHTS.items()
        )
        return min(100, max(0, round(score * 100)))

    # ── Keyword Fallback ──────────────────────────────────────────────────────

    def _keyword_fallback(self, text: str) -> dict:
        """
        Simple heuristic classifier used when the model is unavailable.
        Mirrors the 6-label output format exactly.
        """
        t = text.lower()

        lexicon = {
            "Joy":     ["happy","great","wonderful","excited","joy","love","amazing",
                         "good","fantastic","delighted","glad","cheerful","thrilled"],
            "Calm":    ["calm","peaceful","relaxed","serene","content","tranquil",
                         "mindful","okay","fine","alright","balanced"],
            "Sadness": ["sad","unhappy","depressed","lonely","hopeless","miserable",
                         "cry","grief","heartbroken","down","gloomy","disappointed"],
            "Stress":  ["stress","overwhelmed","exhausted","tired","pressure",
                         "deadline","burnout","hectic","frantic","drained"],
            "Anger":   ["angry","furious","rage","hate","mad","irritated","annoyed",
                         "frustrated","resentful","bitter","hostile"],
            "Fear":    ["afraid","scared","anxious","worried","nervous","dread",
                         "panic","terrified","uneasy","apprehensive","phobia"],
        }

        raw = {em: max(sum(1 for w in words if w in t), 0.05) for em, words in lexicon.items()}
        total = sum(raw.values())

        emotions = sorted(
            [{"emotion": k, "percentage": round((v / total) * 100, 1)} for k, v in raw.items()],
            key=lambda x: x["percentage"], reverse=True,
        )
        dominant      = emotions[0]["emotion"]
        wellness_index = self._wellness(raw, total)

        return {"emotions": emotions, "dominant": dominant, "wellnessIndex": wellness_index}
