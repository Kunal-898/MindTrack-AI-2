"""
claude_service.py — Claude Enrichment Service
Receives DistilRoBERTa emotion scores and enriches them with:
  • Trigger keywords extracted from the journal text
  • 3 personalised, actionable wellness suggestions
  • A concise one-sentence emotional summary

Claude only handles language understanding — all emotion scores
come from the DistilRoBERTa model.
"""

import os
import json
import logging

logger = logging.getLogger("mindtrack.claude")

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

SYSTEM_PROMPT = """You are a compassionate mental wellness assistant for MindTrack.
You receive a journal entry plus its emotion analysis (produced by a DistilRoBERTa model).
Your only job is to return a JSON object with three fields:

{
  "triggers":    ["word_or_phrase_1", "word_or_phrase_2", "word_or_phrase_3"],
  "suggestions": ["actionable_tip_1", "actionable_tip_2", "actionable_tip_3"],
  "summary":     "One empathetic sentence summarising the emotional state."
}

Rules:
- triggers: 3 to 5 actual words or short phrases from the journal text that carry emotional weight
- suggestions: exactly 3 specific, practical mental-wellness tips tailored to the dominant emotion and wellness index
- summary: exactly one sentence, warm and non-judgmental
- Return ONLY the JSON object — no markdown fences, no explanation"""


class ClaudeEnrichmentService:
    """
    Calls Anthropic Claude (claude-sonnet-4-20250514) to enrich
    the DistilRoBERTa emotion output with human-quality language.

    Gracefully degrades to a rule-based fallback if:
      - ANTHROPIC_API_KEY is not set
      - The anthropic package is not installed
      - The API call fails for any reason
    """

    def __init__(self):
        self.client    = None
        self.available = False
        self._init()

    def _init(self):
        if not ANTHROPIC_API_KEY:
            logger.warning("ANTHROPIC_API_KEY not set — Claude enrichment disabled, using fallback")
            return
        try:
            import anthropic
            self.client    = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
            self.available = True
            logger.info("✅ Claude enrichment service ready")
        except ImportError:
            logger.error("anthropic package missing — run: pip install anthropic")
        except Exception as e:
            logger.error(f"Claude init error: {e}")

    # ── Public API ────────────────────────────────────────────────────────────

    def enrich(self, text: str, model_result: dict) -> dict:
        """
        Enrich model_result with Claude-generated language insights.

        Args:
            text         : original journal entry string
            model_result : output of EmotionClassifier.predict()
                           keys: emotions, dominant, wellnessIndex

        Returns:
            dict with keys: triggers (list), suggestions (list), summary (str)
        """
        if not self.available:
            return self._fallback(text, model_result)

        try:
            dominant      = model_result.get("dominant", "Calm")
            wellness      = model_result.get("wellnessIndex", 50)
            top_emotions  = model_result.get("emotions", [])[:3]
            emotion_lines = "\n".join(
                f"  - {e['emotion']}: {e['percentage']}%"
                for e in top_emotions
            )

            user_content = (
                f"Journal entry:\n\"{text}\"\n\n"
                f"DistilRoBERTa emotion analysis:\n"
                f"  Dominant emotion : {dominant}\n"
                f"  Wellness index   : {wellness}/100\n"
                f"  Top emotions:\n{emotion_lines}"
            )

            message = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=400,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_content}],
            )

            raw   = "".join(b.text for b in message.content if hasattr(b, "text"))
            clean = raw.replace("```json", "").replace("```", "").strip()
            data  = json.loads(clean)

            return {
                "triggers":    data.get("triggers", [])[:5],
                "suggestions": data.get("suggestions", [])[:3],
                "summary":     data.get("summary", "").strip(),
            }

        except json.JSONDecodeError as e:
            logger.error(f"Claude returned invalid JSON: {e}")
            return self._fallback(text, model_result)
        except Exception as e:
            logger.error(f"Claude enrichment error: {e}")
            return self._fallback(text, model_result)

    # ── Fallback ──────────────────────────────────────────────────────────────

    def _fallback(self, text: str, model_result: dict) -> dict:
        """
        Rule-based enrichment used when Claude is unavailable.
        Produces reasonable triggers, suggestions, and a summary
        from the model result without any external API call.
        """
        dominant = model_result.get("dominant", "Calm")
        wellness = model_result.get("wellnessIndex", 50)

        # ── Triggers: top non-stopword tokens by frequency ────────────────────
        STOPWORDS = {
            "i","me","my","the","a","an","and","or","but","in","on","at","to",
            "for","of","is","it","that","this","was","am","are","be","been",
            "have","has","had","do","did","will","would","could","should","not",
            "no","with","we","they","he","she","you","so","just","really","very",
            "feel","felt","today","day","time","when","what","how","all","some",
            "about","then","than","there","here","if","can","get","got","now",
            "like","also","more","much","many","one","two","three","its","his",
            "her","our","your","their","from","into","over","after","before",
        }
        tokens = [
            w.strip(".,!?;:'\"()-").lower()
            for w in text.split()
            if len(w.strip(".,!?;:'\"()-")) > 3
        ]
        freq: dict[str, int] = {}
        for w in tokens:
            if w not in STOPWORDS:
                freq[w] = freq.get(w, 0) + 1
        triggers = [w for w, _ in sorted(freq.items(), key=lambda x: x[1], reverse=True)[:5]]
        if not triggers:
            triggers = ["today", "feeling", "moment"]

        # ── Suggestions: emotion-specific templates ───────────────────────────
        suggestion_map = {
            "Joy": [
                "Savour this positive feeling — write down exactly what made you happy.",
                "Share your joy with someone close to you today.",
                "Use this energy to tackle something you've been putting off.",
            ],
            "Calm": [
                "Maintain this balance with 5 minutes of deep breathing before bed.",
                "Use this clarity to plan tomorrow's priorities mindfully.",
                "Practise gratitude journaling to anchor this peaceful state.",
            ],
            "Sadness": [
                "Reach out to a trusted friend or family member — connection heals.",
                "Allow yourself to feel without judgement; this too shall pass.",
                "A gentle walk in nature can shift your perspective significantly.",
            ],
            "Stress": [
                "Break your tasks into three small, immediately actionable steps.",
                "Try box breathing: inhale 4s → hold 4s → exhale 4s → hold 4s.",
                "Schedule a 10-minute no-screen break within the next hour.",
            ],
            "Anger": [
                "Try progressive muscle relaxation — tense and release each muscle group.",
                "Write the full unfiltered version of your frustration, then put it aside.",
                "A brisk 15-minute walk helps metabolise stress hormones naturally.",
            ],
            "Fear": [
                "Ground yourself with 5-4-3-2-1: name 5 things you can see right now.",
                "Challenge the fear: ask yourself — what is the realistic probability?",
                "Talk to someone you trust about what is worrying you today.",
            ],
        }
        suggestions = suggestion_map.get(dominant, suggestion_map["Calm"])

        # ── Summary ───────────────────────────────────────────────────────────
        level = (
            "positive and energised" if wellness >= 65
            else "moderately balanced" if wellness >= 40
            else "challenging and emotionally heavy"
        )
        summary = (
            f"Your journal reflects a {level} emotional state, "
            f"with {dominant.lower()} as the dominant feeling "
            f"and a wellness index of {wellness} out of 100."
        )

        return {"triggers": triggers, "suggestions": suggestions, "summary": summary}
