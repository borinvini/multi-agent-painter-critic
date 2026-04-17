# Multi-Agent Painter & Critic — Design Spec

**Date:** 2026-04-16
**Assignment:** AI Engineering — Assignment 3

---

## Overview

A two-agent system built with the AG2 framework in which a Painter agent iteratively draws on a 200×200 pixel canvas and a Critic agent evaluates the visual output each round. The subject prompt is configurable at runtime. The system runs for a configurable number of rounds (default 10), saving one image per round.

---

## Architecture & Conversation Flow

**Pattern:** Two-agent sequential chat via `initiate_chat`.

The Painter initiates the conversation by sending the first message (which includes the subject prompt and the initial blank canvas image). The Critic replies with structured visual feedback. This alternates for 10 rounds.

**Stop mechanism:** `max_consecutive_auto_reply` is set on both agents:
- `critic.max_consecutive_auto_reply = NUM_ROUNDS` — primary stop: Critic stops after 10 rounds of feedback.
- `painter.max_consecutive_auto_reply = NUM_ROUNDS` — safety stop.

For `NUM_ROUNDS=10`, the conversation shape is:
- Painter sends initial message (round 1 draw) — not an auto-reply
- Critic auto-replies ×10 (rounds 1–10 critique)
- Painter auto-replies ×9 (rounds 2–10 draw)
- Critic hits `max_consecutive_auto_reply=10` → AG2 terminates the chat

**Image injection — both agents see the canvas:**

Two `register_reply` hooks are registered, one per agent:

- **Critic's hook:** fires before the Critic's LLM is called. Reads the current canvas (just drawn by the Painter), encodes it as base64 PNG, and prepends it to the incoming message in OpenAI vision format so the Critic evaluates the actual visual output.
- **Painter's hook:** fires before the Painter's LLM is called (on rounds 2–10, when it receives the Critic's feedback). Reads the current canvas and prepends it the same way so the Painter can see where it left off before deciding what to draw next.

Both hooks inject the image in the same OpenAI vision format:

```python
[
  {"type": "image_url", "image_url": {"url": "data:image/png;base64,<...>"}},
  {"type": "text", "text": "<incoming message text>"}
]
```

This ensures both agents always see the latest visual state of the canvas without relying on the LLM to call a tool.

**Round tracking & image saving:** A module-level round counter is incremented inside the Critic's `register_reply` hook (before the Critic responds), and the canvas is saved as `output/round_NN.png` at that point. This guarantees one saved image per completed Painter turn.

---

## Canvas & Drawing Tools

**Canvas:** A `PIL.Image` object (200×200, RGB, white background) stored as a module-level shared variable. All tool functions read from and write to this shared object.

**Three tools registered on the Painter:**

| Tool | Signature | Purpose |
|------|-----------|---------|
| `draw_pixels` | `(pixels: list[dict])` — each dict: `{x, y, r, g, b}` | Batch-draw individual pixels for fine detail and precise placement |
| `draw_line` | `(x1, y1, x2, y2, r, g, b, width=2)` | Draw a straight line — outlines, stems, horizon lines |
| `draw_filled_rectangle` | `(x, y, width, height, r, g, b)` | Fill a rectangular region — sky, ground, large color blocks |

Each tool validates coordinates (clamped to 0–199) and color values (clamped to 0–255). After every Painter turn, the canvas is saved to disk.

**Image encoding utility:** `canvas_to_base64() -> str` encodes the current PIL image as a PNG base64 string. Called inside the Critic's `register_reply` hook.

---

## Agent System Messages

**Painter:**
```
You are a Painter agent. Your job is to draw on a 200x200 pixel canvas using your drawing tools.
You are drawing: {SUBJECT_PROMPT}

Each turn you MUST call drawing tools to add or refine elements on the canvas.
Draw multiple pixels/shapes per turn — single pixels produce no visible progress.
After drawing, briefly describe what you drew and what you plan to improve next.
When you receive feedback from the Critic, use it to guide your next drawing actions.
```

**Critic:**
```
You are an art Critic agent. Each turn you receive an image of the current canvas and evaluate it.
The subject being drawn is: {SUBJECT_PROMPT}

Provide structured feedback with three parts:
1. What works well (be specific about visual elements)
2. What should be changed or is missing
3. Concrete suggestions for the next round (specific colors, positions, shapes)

Be constructive and actionable. The Painter will use your feedback to improve the drawing.
```

---

## LLM Configuration

Both agents use the same LLM config:

```python
llm_config = {
    "config_list": [{
        "model": "openai/gpt-4.1-mini",
        "base_url": "https://5f5832nb90.execute-api.eu-central-1.amazonaws.com/v1",
        "api_key": "none",
    }],
    "cache_seed": None,  # disable caching so each round gets a fresh response
}
```

---

## File Structure

```
Multi-Agent Painter & Critic/
├── painter_critic.py       # main script (single file)
├── README.md               # assignment-required documentation
├── requirements.txt        # ag2[openai], pillow
└── output/                 # created at runtime
    ├── round_01.png
    ├── round_02.png
    ├── ...
    ├── round_10.png
    └── conversation_log.txt
```

---

## CLI Interface

```bash
python painter_critic.py --subject "a sunset over the ocean" --rounds 10
```

- `--subject` (required): the drawing prompt passed into both agents' system messages
- `--rounds` (optional, default 10): number of Painter→Critic iterations

---

## Conversation Log

At the end of `initiate_chat`, the full message history from the chat result is written to `output/conversation_log.txt`, with agent name and round number prefixed to each message.

---

## Key Design Decisions

1. **`register_reply` for image injection** — more reliable than giving agents a `get_canvas` tool, since it doesn't depend on the LLM remembering to call the tool every round.
2. **Pixel-level tools** — `draw_pixels`, `draw_line`, `draw_filled_rectangle` give the Painter fine-to-coarse control without requiring it to reason about shapes abstractly.
3. **Shared module-level canvas** — avoids serialization complexity; tools directly mutate the PIL image.
4. **Single script** — keeps the submission simple and self-contained; no package structure needed.
5. **`cache_seed=None`** — disables AG2's response caching so each round produces a genuine LLM response.
