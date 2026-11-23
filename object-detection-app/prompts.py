"""
Prompt templates for the safety monitor.

Why a separate module:
  • Prompts are code — they should be diffable, versioned, and testable
  • Changing a prompt shouldn't require touching the client code
  • Testing different prompts is trivial: just pass a different string
"""

# ---------------------------------------------------------------------------
# Helmet check — the only prompt needed for the current use case.
# ---------------------------------------------------------------------------
HELMET_CHECK_PROMPT = """\
You are a workplace safety inspector analyzing a single annotated frame from a
construction site camera. The frame has bounding boxes drawn around detected
persons, each labeled with a track ID of the form #N.

TASK:
For each person in a bounding box, determine whether they are wearing a
SAFETY HELMET (hard hat).

VISUAL CRITERIA FOR A HELMET:
  • Rigid, rounded shell; usually a visible brim or peak.
  • Often high-visibility (yellow, white, orange, or red).
  • May have a chin strap, logo, or ridge lines.
  • A bare head shows hair, skin, or a soft cloth cap — none of these count.

WHEN YOU CANNOT TELL, DO NOT GUESS:
  • Head cropped by the frame edge         → helmet = null
  • Head turned away or heavily occluded   → helmet = null
  • Too dark or blurry to see the head     → helmet = null
  • In these cases confidence MUST be < 0.5.

UNSAFE RULE:
  • Set "unsafe": true if at least one person has helmet=false with
    confidence >= 0.6.
  • Otherwise set "unsafe": false.

RETURN ONLY A JSON OBJECT matching this schema. No prose, no markdown fences,
no extra fields:
{
  "unsafe": <bool>,
  "summary": "<one sentence, max 120 chars>",
  "persons": [
    {
      "track_id": <int or null>,
      "helmet": <true | false | null>,
      "confidence": <float 0.0-1.0>,
      "notes": "<short reason>"
    }
  ]
}
"""