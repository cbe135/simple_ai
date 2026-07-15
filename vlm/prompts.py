"""Prompt construction and answer parsing for VQA-style medical classification.

The VLM "classifies" by generating the class as text. We build a prompt from a
template (with ``{modality}`` / ``{condition}`` substituted) and map the
generated text back to an integer label using the config's ``label_map``.
"""

from __future__ import annotations


def format_prompt(template: str, modality: str, condition: str) -> str:
    """Fill {modality}/{condition} placeholders; return template unchanged if absent."""
    try:
        return str(template).format(modality=modality, condition=condition)
    except (KeyError, IndexError, ValueError):
        return str(template)


def build_messages(prompt_text: str, target_text: str | None = None):
    """Build a Gemma3-style chat conversation for one image+prompt sample.

    ``target_text`` is the assistant answer used for training; omit it for inference.
    """
    user_content = [{"type": "image"}, {"type": "text", "text": prompt_text}]
    messages = [{"role": "user", "content": user_content}]
    if target_text is not None:
        messages.append({"role": "assistant", "content": target_text})
    return messages


def parse_prediction(text: str, label_map: dict) -> int | None:
    """Map generated ``text`` back to an integer label via ``label_map``.

    Strategy: exact match, then substring containment, then a binary yes/no
    heuristic. Returns ``None`` when nothing matches (treated as "unknown").
    """
    if text is None:
        return None
    text_l = text.lower().strip()
    if not text_l:
        return None

    # 1) exact match
    for idx, ans in label_map.items():
        if text_l == str(ans).lower().strip():
            return int(idx)

    # 2) substring containment (longest answer first to avoid partial clashes)
    best = None
    for idx, ans in sorted(label_map.items(), key=lambda kv: -len(str(kv[1]))):
        if str(ans).lower() in text_l:
            best = int(idx)
            break
    if best is not None:
        return best

    # 3) binary yes/no heuristic
    yes_idx = next((int(i) for i, a in label_map.items() if str(a).lower() == "yes"), None)
    no_idx = next((int(i) for i, a in label_map.items() if str(a).lower() == "no"), None)
    if "yes" in text_l and yes_idx is not None:
        return yes_idx
    if "no" in text_l and no_idx is not None:
        return no_idx
    return None
