# Drawing Quality Improvements Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prevent drawing quality degradation across rounds by adding two new shape tools, restructuring the Painter with phase-based rules, and restructuring the Critic with PRESERVE/IMPROVE zone feedback.

**Architecture:** Two new drawing functions (`draw_circle`, `draw_polygon`) are inserted into `painter_critic.py` alongside the existing three tools and registered on both agents. `build_agents` is updated to compute phase boundaries from `num_rounds` and use them in rewritten Painter and Critic system messages.

**Tech Stack:** Python 3, AG2 (`autogen`), Pillow (`PIL.ImageDraw`), pytest

---

## File Map

| File | What changes |
|------|-------------|
| `painter_critic.py` | Add `draw_circle` and `draw_polygon` functions after `draw_filled_rectangle` (line 75). In `build_agents`: compute phase boundaries, register 2 new tools on both agents, replace both system messages. |
| `tests/test_tools.py` | Add `TestDrawCircle` (3 tests) and `TestDrawPolygon` (3 tests) classes. |

No other files change.

---

## Task 1: `draw_circle` — function and tests

**Files:**
- Modify: `painter_critic.py` (insert after line 75)
- Modify: `tests/test_tools.py` (append new test class)

- [ ] **Step 1: Write failing tests for `draw_circle`**

Append to `tests/test_tools.py`:

```python
class TestDrawCircle:
    def test_center_pixel_correct_color(self):
        reset()
        pc.draw_circle(100, 100, 10, 255, 0, 0)
        assert pc.canvas.getpixel((100, 100)) == (255, 0, 0)

    def test_clamps_near_edge_no_error(self):
        reset()
        # Center at (0,0) radius 20: bounding box clips to canvas — no crash
        pc.draw_circle(0, 0, 20, 0, 255, 0)
        assert pc.canvas.getpixel((0, 0)) == (0, 255, 0)

    def test_returns_confirmation_string(self):
        reset()
        result = pc.draw_circle(50, 50, 5, 0, 0, 255)
        assert isinstance(result, str) and "circle" in result.lower()
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_tools.py::TestDrawCircle -v
```

Expected: 3 failures — `AttributeError: module 'painter_critic' has no attribute 'draw_circle'`

- [ ] **Step 3: Implement `draw_circle` in `painter_critic.py`**

Insert after the `draw_filled_rectangle` function (after line 75, before the blank lines leading to `canvas_to_base64`):

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

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_tools.py::TestDrawCircle -v
```

Expected: 3 passed

- [ ] **Step 5: Run full test suite to check no regressions**

```bash
pytest tests/test_tools.py -v
```

Expected: all previously passing tests still pass

- [ ] **Step 6: Commit**

```bash
git add painter_critic.py tests/test_tools.py
git commit -m "feat: add draw_circle tool"
```

---

## Task 2: `draw_polygon` — function and tests

**Files:**
- Modify: `painter_critic.py` (insert after `draw_circle`)
- Modify: `tests/test_tools.py` (append new test class)

- [ ] **Step 1: Write failing tests for `draw_polygon`**

Append to `tests/test_tools.py`:

```python
class TestDrawPolygon:
    def test_interior_pixel_correct_color(self):
        reset()
        # Triangle: top-center (100,50), bottom-left (50,150), bottom-right (150,150)
        # Point (100, 120) is inside this triangle
        pc.draw_polygon(
            [{"x": 100, "y": 50}, {"x": 50, "y": 150}, {"x": 150, "y": 150}],
            0, 128, 0,
        )
        assert pc.canvas.getpixel((100, 120)) == (0, 128, 0)

    def test_clamps_out_of_bounds_no_error(self):
        reset()
        # Vertices partially outside canvas — should clamp and not crash
        pc.draw_polygon(
            [{"x": -10, "y": -10}, {"x": 250, "y": -10}, {"x": 100, "y": 100}],
            255, 255, 0,
        )
        assert pc.canvas.getpixel((100, 50)) == (255, 255, 0)

    def test_returns_confirmation_string(self):
        reset()
        result = pc.draw_polygon(
            [{"x": 10, "y": 10}, {"x": 50, "y": 10}, {"x": 30, "y": 40}],
            100, 100, 100,
        )
        assert isinstance(result, str) and "polygon" in result.lower()
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_tools.py::TestDrawPolygon -v
```

Expected: 3 failures — `AttributeError: module 'painter_critic' has no attribute 'draw_polygon'`

- [ ] **Step 3: Implement `draw_polygon` in `painter_critic.py`**

Insert after the `draw_circle` function:

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

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_tools.py::TestDrawPolygon -v
```

Expected: 3 passed

- [ ] **Step 5: Run full test suite**

```bash
pytest tests/test_tools.py -v
```

Expected: all tests pass

- [ ] **Step 6: Commit**

```bash
git add painter_critic.py tests/test_tools.py
git commit -m "feat: add draw_polygon tool"
```

---

## Task 3: Update `build_agents` — phase boundaries, tool registration, system prompts

**Files:**
- Modify: `painter_critic.py` (`build_agents` function)

This task makes three related changes inside `build_agents`:
1. Compute phase boundary variables at the top of the function
2. Register `draw_circle` and `draw_polygon` on both agents
3. Replace both system messages

- [ ] **Step 1: Add phase boundary calculation at the top of `build_agents`**

In `build_agents`, immediately after the closing `"""` of the docstring and before the `llm_config = {` line, insert:

```python
    # Phase boundaries (proportional to num_rounds: ~30% / 40% / 30%)
    p1_end = max(1, num_rounds * 3 // 10)
    p2_start = p1_end + 1
    p2_end = max(p2_start, num_rounds * 7 // 10)
    p3_start = p2_end + 1
```

- [ ] **Step 2: Verify phase boundaries for num_rounds=10**

Open a Python shell and confirm:
```python
num_rounds = 10
p1_end = max(1, num_rounds * 3 // 10)   # → 3
p2_start = p1_end + 1                    # → 4
p2_end = max(p2_start, num_rounds * 7 // 10)  # → 7
p3_start = p2_end + 1                    # → 8
print(p1_end, p2_start, p2_end, p3_start)  # 3 4 7 8
```

- [ ] **Step 3: Replace the Painter system message**

Replace the entire `system_message=(...)` argument in the `painter = ConversableAgent(...)` call with:

```python
        system_message=(
            f"You are a Painter agent. Your job is to draw on a {CANVAS_SIZE}x{CANVAS_SIZE} pixel canvas "
            f"using your drawing tools.\nYou are drawing: {subject}\n\n"
            "DRAWING PHASES — the Critic announces the current round at the start of each message. "
            "Follow these rules strictly based on the round number:\n\n"
            f"PHASE 1 — Structure (rounds 1-{p1_end}):\n"
            "  Allowed tools: draw_filled_rectangle, draw_circle, draw_polygon\n"
            "  Goal: establish the full composition with major color blocks. Cover all key areas "
            f"of the subject. Do not leave large white regions by the end of round {p1_end}.\n\n"
            f"PHASE 2 — Detail (rounds {p2_start}-{p2_end}):\n"
            "  Allowed tools: draw_pixels, draw_line\n"
            "  FORBIDDEN: draw_filled_rectangle, draw_circle, draw_polygon over any area that "
            "already contains non-white pixels.\n"
            "  Goal: add texture, shading, and fine detail on top of the Phase 1 structure.\n\n"
            f"PHASE 3 — Polish (rounds {p3_start}-{num_rounds}):\n"
            "  Allowed tools: draw_pixels only\n"
            "  Goal: fix specific pixel-level issues the Critic identifies. Do not change anything "
            "that is not listed in the Critic's IMPROVE section.\n\n"
            "NON-DESTRUCTIVE RULE (all phases):\n"
            "  Never use a filled shape to cover an area that already has non-white pixels unless "
            "explicitly correcting a mistake identified by the Critic.\n\n"
            "BATCHING RULE:\n"
            "  Always call 3-5 tools per message. Never call just one and wait.\n"
            "  After tool calls, briefly state what you drew and which phase you are in."
        ),
```

- [ ] **Step 4: Replace the Critic system message**

Replace the entire `system_message=(...)` argument in the `critic = ConversableAgent(...)` call with:

```python
        system_message=(
            f"You are an art Critic agent. Each turn you receive an image of the current canvas "
            f"and evaluate it.\nThe subject being drawn is: {subject}\n"
            f"Total rounds: {num_rounds}\n\n"
            "ROUND TRACKING: Count your replies starting from 1. Begin EVERY message with:\n"
            f"  === Round N of {num_rounds} | Phase: [Structure / Detail / Polish] ===\n\n"
            "PHASES:\n"
            f"  Structure: rounds 1-{p1_end}          — broad shapes, major color blocks\n"
            f"  Detail:    rounds {p2_start}-{p2_end} — texture, shading, fine work\n"
            f"  Polish:    rounds {p3_start}-{num_rounds} — pixel-level fixes only\n\n"
            "FEEDBACK FORMAT — use this exact structure every round:\n\n"
            f"  === Round N of {num_rounds} | Phase: [name] ===\n\n"
            "  PRESERVE (do not touch):\n"
            "  - [Established area + approximate bounding box, e.g. 'Trunk: x 85-115, y 150-190']\n\n"
            "  IMPROVE (this round only — maximum 3 items):\n"
            "  1. [Specific change with region, colors, and tool suggestion]\n"
            "  2. [Specific change with region, colors, and tool suggestion]\n\n"
            "  PHASE NOTE: [One sentence on what tool calls are appropriate this round]\n\n"
            "RULES:\n"
            "  - Never ask the Painter questions. Give instructions only.\n"
            "  - Never suggest changes to PRESERVE zones.\n"
            "  - Maximum 3 items in IMPROVE per round — pick the highest-impact changes.\n"
            "  - If an improvement from the previous round was completed well, move it to PRESERVE.\n"
            "  - Be specific: name colors as RGB values, regions as coordinate ranges."
        ),
```

- [ ] **Step 5: Register `draw_circle` on Painter (caller + executor)**

After the `register_function(draw_filled_rectangle, ...)` block (around line 219), add:

```python
    register_function(
        draw_circle,
        caller=painter,
        executor=painter,
        name="draw_circle",
        description=(
            "Draw a filled circle on the canvas. "
            "Parameters: cx (int 0-199, center x), cy (int 0-199, center y), "
            "radius (int, radius in pixels), r (int 0-255), g (int 0-255), b (int 0-255)."
        ),
    )
```

- [ ] **Step 6: Register `draw_polygon` on Painter (caller + executor)**

Immediately after the `draw_circle` registration, add:

```python
    register_function(
        draw_polygon,
        caller=painter,
        executor=painter,
        name="draw_polygon",
        description=(
            "Draw a filled polygon on the canvas. "
            "Parameters: points (list of dicts with keys x (int 0-199) and y (int 0-199) "
            "defining the polygon vertices), r (int 0-255), g (int 0-255), b (int 0-255). "
            "Use for triangles, irregular shapes, and any non-rectangular filled areas."
        ),
    )
```

- [ ] **Step 7: Register `draw_circle` and `draw_polygon` on Critic (executor only)**

After the existing `critic.register_for_execution(name="draw_filled_rectangle")(draw_filled_rectangle)` line, add:

```python
    critic.register_for_execution(name="draw_circle")(draw_circle)
    critic.register_for_execution(name="draw_polygon")(draw_polygon)
```

- [ ] **Step 8: Run full test suite to confirm nothing is broken**

```bash
pytest tests/test_tools.py -v
```

Expected: all tests pass (the system message changes are not unit-tested but the tool registrations do not affect existing tests)

- [ ] **Step 9: Commit**

```bash
git add painter_critic.py
git commit -m "feat: phase-based prompts, zone feedback, register circle and polygon tools"
```
