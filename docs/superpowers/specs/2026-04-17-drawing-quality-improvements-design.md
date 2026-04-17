# Drawing Quality Improvements — Design Spec

**Date:** 2026-04-17
**Project:** Multi-Agent Painter & Critic

---

## Overview

The current system produces drawings that degrade over rounds: the Painter over-paints established areas with filled rectangles, the Critic repeats the same feedback indefinitely, and neither agent has a clear sense of what has been completed vs. what still needs work. This spec defines targeted improvements — phase-based drawing rules, zone-based Critic feedback, and two new shape tools — that make each round visibly improve on the previous one.

---

## Problem Summary

Two failure modes observed in output images and conversation logs:

1. **Destructive over-painting**: The Painter uses `draw_filled_rectangle` in later rounds to "reset" areas that already contain pixel-level detail, then re-adds detail on top of the flattened surface. This creates a noisy, inconsistent texture rather than genuine refinement.

2. **Repetitive feedback loop**: The Critic repeats the same suggestions round after round ("add texture", "soften edges") without acknowledging what was already completed. The Painter attempts the same changes again, accumulating noise rather than making targeted improvements.

---

## Changes

### 1. Two New Drawing Tools

**`draw_circle(cx, cy, radius, r, g, b)`**

Draws a filled circle using PIL's `ellipse`. Coordinates and color values are clamped. Use cases: round tree canopy, sun, moon, rounded shapes. Replaces the current pattern of approximating circles with many `draw_pixels` calls.

```python
def draw_circle(
    cx: Annotated[int, "Center x (0-199)"],
    cy: Annotated[int, "Center y (0-199)"],
    radius: Annotated[int, "Radius in pixels"],
    r: Annotated[int, "Red 0-255"],
    g: Annotated[int, "Green 0-255"],
    b: Annotated[int, "Blue 0-255"],
) -> str:
    """Draw a filled circle on the canvas."""
    draw = ImageDraw.Draw(canvas)
    x0 = max(0, cx - radius)
    y0 = max(0, cy - radius)
    x1 = min(CANVAS_SIZE - 1, cx + radius)
    y1 = min(CANVAS_SIZE - 1, cy + radius)
    draw.ellipse(
        [(x0, y0), (x1, y1)],
        fill=(max(0, min(255, r)), max(0, min(255, g)), max(0, min(255, b))),
    )
    return f"Drew circle at ({cx},{cy}) radius {radius}."
```

**`draw_polygon(points, r, g, b)`**

Draws a filled polygon from a list of `{x, y}` vertex dicts. Coordinates are clamped. Use cases: triangular pine tree canopy, irregular cloud, mountain silhouette. More expressive than stacking rectangles for non-rectangular shapes.

```python
def draw_polygon(
    points: Annotated[list[dict], "List of dicts with keys x (0-199) and y (0-199) defining polygon vertices"],
    r: Annotated[int, "Red 0-255"],
    g: Annotated[int, "Green 0-255"],
    b: Annotated[int, "Blue 0-255"],
) -> str:
    """Draw a filled polygon on the canvas."""
    draw = ImageDraw.Draw(canvas)
    coords = [
        (max(0, min(CANVAS_SIZE - 1, int(p["x"]))),
         max(0, min(CANVAS_SIZE - 1, int(p["y"]))))
        for p in points
    ]
    draw.polygon(
        coords,
        fill=(max(0, min(255, r)), max(0, min(255, g)), max(0, min(255, b))),
    )
    return f"Drew polygon with {len(points)} vertices."
```

Both tools are registered on both agents — Painter as caller+executor, Critic as executor-only — matching the existing pattern.

---

### 2. Phase-Based Painter System Prompt

Phase boundaries are calculated from `num_rounds` in `build_agents`:

```python
p1_end = max(1, num_rounds * 3 // 10)        # ~30%
p2_start = p1_end + 1
p2_end = max(p2_start, num_rounds * 7 // 10)  # ~70%
p3_start = p2_end + 1
```

For `num_rounds=10`: p1_end=3, p2_start=4, p2_end=7, p3_start=8.

**Painter system message:**

```
You are a Painter agent. Your job is to draw on a {CANVAS_SIZE}x{CANVAS_SIZE} pixel canvas
using your drawing tools.
You are drawing: {subject}

DRAWING PHASES — the Critic announces the current round. Follow these rules strictly:

PHASE 1 — Structure (rounds 1–{p1_end}):
  Allowed tools: draw_filled_rectangle, draw_circle, draw_polygon
  Goal: establish the full composition with major color blocks. Cover all key areas
  of the subject. Do not leave large white regions by the end of round {p1_end}.

PHASE 2 — Detail (rounds {p2_start}–{p2_end}):
  Allowed tools: draw_pixels, draw_line
  FORBIDDEN: draw_filled_rectangle, draw_circle, draw_polygon over any area that
  already contains non-white pixels.
  Goal: add texture, shading, and fine detail on top of the Phase 1 structure.

PHASE 3 — Polish (rounds {p3_start}–{num_rounds}):
  Allowed tools: draw_pixels only
  Goal: fix specific pixel-level issues the Critic identifies. Do not change anything
  that is not listed in the Critic's IMPROVE section.

NON-DESTRUCTIVE RULE (all phases):
  Never use a filled shape to cover an area that already has non-white pixels unless
  explicitly correcting a mistake identified by the Critic.

BATCHING RULE:
  Always call 3–5 tools per message. Never call just one and wait.
  After tool calls, briefly state what you drew and which phase you are in.
```

---

### 3. PRESERVE/IMPROVE Critic System Prompt

Phase boundaries use the same `p1_end`, `p2_end`, `p3_start` values computed in `build_agents`.

**Critic system message:**

```
You are an art Critic agent. Each turn you receive an image of the current canvas
and evaluate it.
The subject being drawn is: {subject}
Total rounds: {num_rounds}

ROUND TRACKING: Count your replies starting from 1. Begin EVERY message with:
  === Round N of {num_rounds} | Phase: [Structure / Detail / Polish] ===

PHASES:
  Structure: rounds 1–{p1_end}         — broad shapes, major color blocks
  Detail:    rounds {p2_start}–{p2_end} — texture, shading, fine work
  Polish:    rounds {p3_start}–{num_rounds} — pixel-level fixes only

FEEDBACK FORMAT — use this exact structure every round:

  === Round N of {num_rounds} | Phase: [name] ===

  PRESERVE (do not touch):
  - [Established area + approximate bounding box, e.g. "Trunk: x 85–115, y 150–190"]
  (One line per area that is working well and should not be repainted)

  IMPROVE (this round only — maximum 3 items):
  1. [Specific change with region, colors, and tool suggestion]
  2. [Specific change with region, colors, and tool suggestion]
  3. [Specific change with region, colors, and tool suggestion]

  PHASE NOTE: [One sentence on what tool calls are appropriate this round]

RULES:
  - Never ask the Painter questions. Give instructions only.
  - Never suggest changes to PRESERVE zones.
  - Maximum 3 items in IMPROVE per round — pick the highest-impact changes.
  - If an improvement from the previous round was completed well, move it to PRESERVE.
  - Be constructive and specific: name colors as RGB values, regions as coordinate ranges.
```

---

## File Changes

| File | Change |
|------|--------|
| `painter_critic.py` | Add `draw_circle`, `draw_polygon` functions |
| `painter_critic.py` | Register both tools on Painter (caller+executor) and Critic (executor) |
| `painter_critic.py` | Compute phase boundaries `p1_end`, `p2_end`, `p3_start` in `build_agents` |
| `painter_critic.py` | Replace Painter system message with phase-based version |
| `painter_critic.py` | Replace Critic system message with PRESERVE/IMPROVE version |
| `tests/test_tools.py` | Add tests for `draw_circle` and `draw_polygon` |

No changes to CLI interface, canvas size, round-saving logic, or image injection hooks.

---

## Key Design Decisions

1. **Phase rules in the Painter's prompt, not enforced in code** — simpler to implement and sufficient given the LLM follows explicit rule lists reliably. Code enforcement would require tracking tool calls per round, adding complexity.

2. **Critic counts rounds itself** — the Critic sees the full conversation history and can count its own replies from 1. This avoids exposing the internal `round_counter` to the agent layer.

3. **Max 3 IMPROVE items** — the primary driver of noise accumulation is the Critic listing 5–6 changes per round and the Painter attempting all of them simultaneously. Capping at 3 forces prioritization.

4. **draw_polygon over draw_triangle** — `draw_polygon` is strictly more expressive (handles triangles as a 3-vertex case) with no added complexity. A dedicated triangle function would be redundant.

5. **Phase boundaries proportional to num_rounds** — 30%/40%/30% split works for any round count, preserving the intent at `--rounds 5` or `--rounds 20`.
