# Multi-Agent Painter & Critic Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a two-agent AG2 system where a Painter draws on a 200×200 canvas over 10 rounds and a Critic evaluates each round's image using vision.

**Architecture:** Critic initiates chat with Painter via `initiate_chat(max_turns=NUM_ROUNDS*2)`. Both agents receive the current canvas via a `process_all_messages_before_reply` hook. The Critic also has a `register_reply` hook at position 0 that saves the canvas as `round_NN.png` before each critique. The Painter executes its own tools (draw_pixels, draw_line, draw_filled_rectangle) on a shared PIL canvas.

**Tech Stack:** Python 3.10+, ag2[openai], Pillow, argparse, pytest

---

## File Map

| File | Responsibility |
|------|---------------|
| `painter_critic.py` | Single-file script: canvas state, tools, hooks, agent builder, CLI |
| `tests/test_tools.py` | Unit tests for canvas tools, utilities, and image injection |
| `requirements.txt` | ag2[openai], Pillow |
| `README.md` | Assignment documentation |
| `output/` | Created at runtime: round_NN.png + conversation_log.txt |

---

## Task 1: Project Scaffold

**Files:**
- Create: `requirements.txt`
- Create: `tests/__init__.py`
- Create: `painter_critic.py` (skeleton only)

- [ ] **Step 1: Create requirements.txt**

```
ag2[openai]
Pillow
```

- [ ] **Step 2: Create tests/__init__.py**

Empty file — makes `tests/` a package so pytest discovers it.

```bash
mkdir tests && touch tests/__init__.py
```

- [ ] **Step 3: Create painter_critic.py skeleton**

```python
# painter_critic.py
import argparse
import base64
import io
import os
from typing import Annotated, Any

from autogen import ConversableAgent, register_function
from PIL import Image, ImageDraw

CANVAS_SIZE = 200
canvas: Image.Image = Image.new("RGB", (CANVAS_SIZE, CANVAS_SIZE), (255, 255, 255))
round_counter: list[int] = [0]
```

- [ ] **Step 4: Install dependencies**

```bash
pip install -r requirements.txt
```

Expected: ag2 and Pillow installed with no errors.

- [ ] **Step 5: Commit**

```bash
git init
git add requirements.txt tests/__init__.py painter_critic.py
git commit -m "feat: project scaffold"
```

---

## Task 2: Canvas Drawing Tools

**Files:**
- Modify: `painter_critic.py` — add draw_pixels, draw_line, draw_filled_rectangle
- Create: `tests/test_tools.py` — tests for all three tools

- [ ] **Step 1: Write failing tests for draw_pixels**

```python
# tests/test_tools.py
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PIL import Image
import painter_critic as pc


def reset():
    pc.canvas = Image.new("RGB", (pc.CANVAS_SIZE, pc.CANVAS_SIZE), (255, 255, 255))
    pc.round_counter[0] = 0


class TestDrawPixels:
    def test_single_pixel_correct_color(self):
        reset()
        pc.draw_pixels([{"x": 10, "y": 20, "r": 255, "g": 0, "b": 0}])
        assert pc.canvas.getpixel((10, 20)) == (255, 0, 0)

    def test_multiple_pixels(self):
        reset()
        pc.draw_pixels([
            {"x": 5, "y": 5, "r": 100, "g": 150, "b": 200},
            {"x": 6, "y": 6, "r": 50, "g": 60, "b": 70},
        ])
        assert pc.canvas.getpixel((5, 5)) == (100, 150, 200)
        assert pc.canvas.getpixel((6, 6)) == (50, 60, 70)

    def test_clamps_out_of_bounds_no_error(self):
        reset()
        pc.draw_pixels([{"x": 300, "y": -5, "r": 255, "g": 255, "b": 255}])
        # Clamped: should not raise

    def test_returns_count_string(self):
        reset()
        result = pc.draw_pixels([{"x": 1, "y": 1, "r": 0, "g": 0, "b": 0}])
        assert "1" in result


class TestDrawLine:
    def test_horizontal_line_midpoint_correct_color(self):
        reset()
        pc.draw_line(10, 50, 100, 50, 0, 255, 0, 1)
        assert pc.canvas.getpixel((55, 50)) == (0, 255, 0)

    def test_returns_confirmation_string(self):
        reset()
        result = pc.draw_line(0, 0, 10, 10, 255, 0, 0)
        assert isinstance(result, str) and len(result) > 0


class TestDrawFilledRectangle:
    def test_fills_interior(self):
        reset()
        pc.draw_filled_rectangle(10, 10, 50, 50, 0, 0, 255)
        assert pc.canvas.getpixel((35, 35)) == (0, 0, 255)

    def test_does_not_affect_outside(self):
        reset()
        pc.draw_filled_rectangle(10, 10, 20, 20, 255, 0, 0)
        assert pc.canvas.getpixel((0, 0)) == (255, 255, 255)
        assert pc.canvas.getpixel((50, 50)) == (255, 255, 255)

    def test_returns_confirmation_string(self):
        reset()
        result = pc.draw_filled_rectangle(0, 0, 10, 10, 0, 0, 0)
        assert isinstance(result, str) and len(result) > 0
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
pytest tests/test_tools.py -v
```

Expected: `AttributeError: module 'painter_critic' has no attribute 'draw_pixels'`

- [ ] **Step 3: Implement draw_pixels, draw_line, draw_filled_rectangle in painter_critic.py**

Add after the module-level canvas declarations:

```python
def draw_pixels(
    pixels: Annotated[list[dict], "List of dicts with keys x (0-199), y (0-199), r, g, b (0-255)"],
) -> str:
    """Batch-draw pixels on the canvas."""
    for p in pixels:
        x = max(0, min(CANVAS_SIZE - 1, int(p["x"])))
        y = max(0, min(CANVAS_SIZE - 1, int(p["y"])))
        r = max(0, min(255, int(p["r"])))
        g = max(0, min(255, int(p["g"])))
        b = max(0, min(255, int(p["b"])))
        canvas.putpixel((x, y), (r, g, b))
    return f"Drew {len(pixels)} pixels."


def draw_line(
    x1: Annotated[int, "Start x (0-199)"],
    y1: Annotated[int, "Start y (0-199)"],
    x2: Annotated[int, "End x (0-199)"],
    y2: Annotated[int, "End y (0-199)"],
    r: Annotated[int, "Red 0-255"],
    g: Annotated[int, "Green 0-255"],
    b: Annotated[int, "Blue 0-255"],
    width: Annotated[int, "Line width in pixels (default 2)"] = 2,
) -> str:
    """Draw a straight line between two points."""
    draw = ImageDraw.Draw(canvas)
    x1 = max(0, min(CANVAS_SIZE - 1, x1))
    y1 = max(0, min(CANVAS_SIZE - 1, y1))
    x2 = max(0, min(CANVAS_SIZE - 1, x2))
    y2 = max(0, min(CANVAS_SIZE - 1, y2))
    draw.line(
        [(x1, y1), (x2, y2)],
        fill=(max(0, min(255, r)), max(0, min(255, g)), max(0, min(255, b))),
        width=max(1, width),
    )
    return f"Drew line ({x1},{y1})\u2192({x2},{y2})."


def draw_filled_rectangle(
    x: Annotated[int, "Top-left x (0-199)"],
    y: Annotated[int, "Top-left y (0-199)"],
    width: Annotated[int, "Width in pixels"],
    height: Annotated[int, "Height in pixels"],
    r: Annotated[int, "Red 0-255"],
    g: Annotated[int, "Green 0-255"],
    b: Annotated[int, "Blue 0-255"],
) -> str:
    """Fill a rectangular region with a solid color."""
    draw = ImageDraw.Draw(canvas)
    x0 = max(0, min(CANVAS_SIZE - 1, x))
    y0 = max(0, min(CANVAS_SIZE - 1, y))
    x1 = max(0, min(CANVAS_SIZE - 1, x + width - 1))
    y1 = max(0, min(CANVAS_SIZE - 1, y + height - 1))
    draw.rectangle(
        [(x0, y0), (x1, y1)],
        fill=(max(0, min(255, r)), max(0, min(255, g)), max(0, min(255, b))),
    )
    return f"Drew rectangle ({x0},{y0}) {width}\xd7{height}."
```

- [ ] **Step 4: Run tests — verify they pass**

```bash
pytest tests/test_tools.py -v
```

Expected: All tests PASS.

- [ ] **Step 5: Commit**

```bash
git add painter_critic.py tests/test_tools.py
git commit -m "feat: implement canvas drawing tools with tests"
```

---

## Task 3: Canvas Utilities (canvas_to_base64, inject_canvas_into_messages, save_canvas)

**Files:**
- Modify: `painter_critic.py` — add three utility functions
- Modify: `tests/test_tools.py` — add utility tests

- [ ] **Step 1: Write failing tests for utilities**

Add to `tests/test_tools.py`:

```python
import base64
import io as _io


class TestCanvasToBase64:
    def test_returns_non_empty_string(self):
        reset()
        result = pc.canvas_to_base64()
        assert isinstance(result, str) and len(result) > 100

    def test_decodes_to_200x200_png(self):
        reset()
        data = base64.b64decode(pc.canvas_to_base64())
        img = Image.open(_io.BytesIO(data))
        assert img.size == (200, 200)
        assert img.mode == "RGB"


class TestInjectCanvas:
    def test_injects_image_url_as_first_content_item(self):
        reset()
        messages = [{"role": "user", "content": "draw a cat"}]
        result = pc.inject_canvas_into_messages(messages)
        assert isinstance(result[-1]["content"], list)
        assert result[-1]["content"][0]["type"] == "image_url"
        assert "base64" in result[-1]["content"][0]["image_url"]["url"]

    def test_preserves_text_as_second_content_item(self):
        reset()
        messages = [{"role": "user", "content": "draw a cat"}]
        result = pc.inject_canvas_into_messages(messages)
        assert result[-1]["content"][1] == {"type": "text", "text": "draw a cat"}

    def test_does_not_touch_earlier_messages(self):
        reset()
        messages = [
            {"role": "user", "content": "earlier"},
            {"role": "user", "content": "latest"},
        ]
        result = pc.inject_canvas_into_messages(messages)
        assert result[0]["content"] == "earlier"

    def test_handles_empty_list(self):
        reset()
        assert pc.inject_canvas_into_messages([]) == []
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
pytest tests/test_tools.py::TestCanvasToBase64 tests/test_tools.py::TestInjectCanvas -v
```

Expected: `AttributeError: module 'painter_critic' has no attribute 'canvas_to_base64'`

- [ ] **Step 3: Implement utilities in painter_critic.py**

Add after the drawing tool functions:

```python
def canvas_to_base64() -> str:
    """Encode the current canvas as a base64 PNG string."""
    buf = io.BytesIO()
    canvas.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def save_canvas(round_num: int) -> None:
    """Save the current canvas to output/round_NN.png."""
    os.makedirs("output", exist_ok=True)
    canvas.save(f"output/round_{round_num:02d}.png")


def inject_canvas_into_messages(messages: list[dict]) -> list[dict]:
    """Hook: prepend the current canvas image to the last message."""
    if not messages:
        return messages
    last = dict(messages[-1])
    text = last.get("content") or ""
    if isinstance(text, str):
        last["content"] = [
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{canvas_to_base64()}"},
            },
            {"type": "text", "text": text},
        ]
    return messages[:-1] + [last]
```

- [ ] **Step 4: Run all tests — verify they pass**

```bash
pytest tests/test_tools.py -v
```

Expected: All tests PASS.

- [ ] **Step 5: Commit**

```bash
git add painter_critic.py tests/test_tools.py
git commit -m "feat: add canvas utilities and image injection hook"
```

---

## Task 4: Critic Round Hook and Agent Builder

**Files:**
- Modify: `painter_critic.py` — add make_critic_round_hook and build_agents

- [ ] **Step 1: Add make_critic_round_hook to painter_critic.py**

Add after the utility functions:

```python
def make_critic_round_hook():
    """
    Returns a register_reply function for the Critic.
    Fires before each Critic LLM call: increments round counter, saves canvas.
    Returns (False, None) to pass control to the next reply handler (the LLM).
    """
    def hook(recipient: ConversableAgent, messages: list, sender: ConversableAgent, config: Any) -> tuple[bool, Any]:
        round_counter[0] += 1
        save_canvas(round_counter[0])
        print(f"\n[Round {round_counter[0]}] Canvas saved as output/round_{round_counter[0]:02d}.png")
        return False, None
    return hook
```

- [ ] **Step 2: Add build_agents function to painter_critic.py**

Add after make_critic_round_hook:

```python
def build_agents(subject: str, num_rounds: int) -> tuple[ConversableAgent, ConversableAgent]:
    """Create and configure the Painter and Critic agents."""
    llm_config = {
        "config_list": [
            {
                "model": "openai/gpt-4.1-mini",
                "base_url": "https://5f5832nb90.execute-api.eu-central-1.amazonaws.com/v1",
                "api_key": "none",
            }
        ],
        "cache_seed": None,
    }

    painter = ConversableAgent(
        name="Painter",
        system_message=(
            f"You are a Painter agent. Your job is to draw on a {CANVAS_SIZE}x{CANVAS_SIZE} pixel canvas "
            f"using your drawing tools.\nYou are drawing: {subject}\n\n"
            "Each turn you MUST call drawing tools to add or refine elements on the canvas. "
            "Draw multiple pixels/shapes per turn — single pixels produce no visible progress.\n"
            "After drawing, briefly describe what you drew and what you plan to improve next.\n"
            "When you receive feedback from the Critic, use it to guide your next drawing actions."
        ),
        llm_config=llm_config,
        human_input_mode="NEVER",
    )

    critic = ConversableAgent(
        name="Critic",
        system_message=(
            f"You are an art Critic agent. Each turn you receive an image of the current canvas and evaluate it.\n"
            f"The subject being drawn is: {subject}\n\n"
            "Provide structured feedback with three parts:\n"
            "1. What works well (be specific about visual elements)\n"
            "2. What should be changed or is missing\n"
            "3. Concrete suggestions for the next round (specific colors, positions, shapes)\n\n"
            "Be constructive and actionable. The Painter will use your feedback to improve the drawing."
        ),
        llm_config=llm_config,
        human_input_mode="NEVER",
        max_consecutive_auto_reply=num_rounds,
    )

    # Register drawing tools: Painter is both the caller (LLM) and executor
    register_function(
        draw_pixels,
        caller=painter,
        executor=painter,
        name="draw_pixels",
        description=(
            "Batch-draw pixels on the canvas. "
            "Input: list of dicts, each with keys: x (int 0-199), y (int 0-199), "
            "r (int 0-255), g (int 0-255), b (int 0-255). "
            "Always provide at least 20 pixels per call for visible progress."
        ),
    )
    register_function(
        draw_line,
        caller=painter,
        executor=painter,
        name="draw_line",
        description=(
            "Draw a straight line on the canvas. "
            "Parameters: x1 (int 0-199), y1 (int 0-199), x2 (int 0-199), y2 (int 0-199), "
            "r (int 0-255), g (int 0-255), b (int 0-255), width (int, default 2)."
        ),
    )
    register_function(
        draw_filled_rectangle,
        caller=painter,
        executor=painter,
        name="draw_filled_rectangle",
        description=(
            "Fill a rectangular region with a solid color. "
            "Parameters: x (int 0-199, top-left), y (int 0-199, top-left), "
            "width (int), height (int), r (int 0-255), g (int 0-255), b (int 0-255)."
        ),
    )

    # Critic: save canvas + inject canvas image before each critique
    critic.register_reply(
        [ConversableAgent],
        make_critic_round_hook(),
        position=0,
    )
    critic.register_hook("process_all_messages_before_reply", inject_canvas_into_messages)

    # Painter: inject current canvas image before each drawing turn
    painter.register_hook("process_all_messages_before_reply", inject_canvas_into_messages)

    return painter, critic
```

- [ ] **Step 3: Smoke-check agent creation (no LLM call)**

```bash
python -c "
import painter_critic as pc
painter, critic = pc.build_agents('a cat', 10)
print('Painter tools:', list(painter._function_map.keys()))
print('Critic max_auto_reply:', critic.max_consecutive_auto_reply)
"
```

Expected output (example):
```
Painter tools: ['draw_pixels', 'draw_line', 'draw_filled_rectangle']
Critic max_auto_reply: 10
```

- [ ] **Step 4: Commit**

```bash
git add painter_critic.py
git commit -m "feat: add agent builder with hooks and tool registration"
```

---

## Task 5: Main Entry Point (run + CLI)

**Files:**
- Modify: `painter_critic.py` — add run(), main()
- Modify: `tests/test_tools.py` — add argparse test

- [ ] **Step 1: Write failing test for CLI argument parsing**

Add to `tests/test_tools.py`:

```python
import argparse


class TestCLI:
    def test_subject_required(self):
        import subprocess, sys
        result = subprocess.run(
            [sys.executable, "painter_critic.py"],
            capture_output=True, text=True
        )
        assert result.returncode != 0
        assert "subject" in result.stderr.lower() or "required" in result.stderr.lower()

    def test_default_rounds_is_10(self):
        # Parse manually using the same argparse setup as main()
        import importlib
        pc = importlib.import_module("painter_critic")
        parser = argparse.ArgumentParser()
        parser.add_argument("--subject", required=True)
        parser.add_argument("--rounds", type=int, default=10)
        args = parser.parse_args(["--subject", "a cat"])
        assert args.rounds == 10
        assert args.subject == "a cat"
```

- [ ] **Step 2: Run test — verify it fails**

```bash
pytest tests/test_tools.py::TestCLI -v
```

Expected: ImportError or test error because main() doesn't exist yet.

- [ ] **Step 3: Add run() and main() to painter_critic.py**

Add at the end of the file (before `if __name__ == "__main__":`):

```python
def run(subject: str, num_rounds: int = 10) -> None:
    """Execute the full Painter–Critic conversation."""
    global canvas, round_counter
    # Reset state for a fresh run
    canvas = Image.new("RGB", (CANVAS_SIZE, CANVAS_SIZE), (255, 255, 255))
    round_counter[0] = 0

    painter, critic = build_agents(subject, num_rounds)

    initial_message = (
        f"Please start drawing: {subject}. "
        f"The canvas is a blank white {CANVAS_SIZE}x{CANVAS_SIZE} image. "
        "Use your drawing tools to begin creating the artwork. "
        "Call at least one drawing tool before responding."
    )

    print(f"Starting {num_rounds}-round Painter & Critic session.")
    print(f"Subject: {subject}\n")

    chat_result = critic.initiate_chat(
        painter,
        message=initial_message,
        max_turns=num_rounds * 2,
    )

    # Save conversation log (text only — strip base64 image content)
    os.makedirs("output", exist_ok=True)
    log_path = "output/conversation_log.txt"
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"Subject: {subject}\nRounds: {num_rounds}\n{'='*60}\n\n")
        for msg in chat_result.chat_history:
            name = msg.get("name") or msg.get("role", "unknown")
            content = msg.get("content") or ""
            if isinstance(content, list):
                # Extract text parts only
                parts = [
                    c["text"]
                    for c in content
                    if isinstance(c, dict) and c.get("type") == "text"
                ]
                content = " ".join(parts)
            if content.strip():
                f.write(f"[{name}]\n{content}\n\n")

    print(f"\nDone. {round_counter[0]} rounds completed.")
    print(f"Images saved in output/ | Log saved to {log_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Multi-Agent Painter & Critic — AG2 demo"
    )
    parser.add_argument("--subject", required=True, help="Drawing subject/prompt")
    parser.add_argument(
        "--rounds",
        type=int,
        default=10,
        help="Number of Painter–Critic rounds (default: 10)",
    )
    args = parser.parse_args()
    run(args.subject, args.rounds)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests — verify they pass**

```bash
pytest tests/test_tools.py -v
```

Expected: All tests PASS.

- [ ] **Step 5: Commit**

```bash
git add painter_critic.py tests/test_tools.py
git commit -m "feat: add run() and main() CLI entry point"
```

---

## Task 6: End-to-End Smoke Test

**Files:**
- Read: `output/round_01.png` (verify it exists and is valid)
- Read: `output/conversation_log.txt` (verify it has content)

- [ ] **Step 1: Run 1-round smoke test**

```bash
python painter_critic.py --subject "a simple red circle on white background" --rounds 1
```

Expected console output:
```
Starting 1-round Painter & Critic session.
Subject: a simple red circle on white background

[Round 1] Canvas saved as output/round_01.png
...
Done. 1 rounds completed.
Images saved in output/ | Log saved to output/conversation_log.txt
```

- [ ] **Step 2: Verify output files exist and are valid**

```bash
python -c "
from PIL import Image
img = Image.open('output/round_01.png')
print('Image size:', img.size)
print('Image mode:', img.mode)

with open('output/conversation_log.txt') as f:
    content = f.read()
print('Log length:', len(content), 'chars')
print('Has Painter:', 'Painter' in content)
print('Has Critic:', 'Critic' in content)
"
```

Expected:
```
Image size: (200, 200)
Image mode: RGB
Log length: <N> chars
Has Painter: True
Has Critic: True
```

- [ ] **Step 3: If image is all white (Painter drew nothing), debug tool registration**

If `round_01.png` is all white, it means the Painter's tool calls weren't executed. Check:

```bash
python -c "
import painter_critic as pc
p, c = pc.build_agents('test', 1)
# Check tool map
print('Function map:', list(p._function_map.keys()))
# Check LLM config has tools
import json
print('LLM tools:', json.dumps(p.llm_config.get('tools', []), indent=2)[:500])
"
```

If `function_map` is empty, the tools weren't registered as executor. Try adding:
```python
# In build_agents, after register_function calls, also manually register for execution:
painter.register_for_execution()(draw_pixels)
painter.register_for_execution()(draw_line)
painter.register_for_execution()(draw_filled_rectangle)
```

- [ ] **Step 4: Run full 10-round session**

```bash
python painter_critic.py --subject "a sunset over the ocean with orange sky and calm water" --rounds 10
```

Expected: `output/round_01.png` through `output/round_10.png` all exist and show progressive artwork.

- [ ] **Step 5: Commit output samples**

```bash
git add output/round_01.png output/round_05.png output/round_10.png output/conversation_log.txt
git commit -m "feat: add sample output from 10-round run"
```

---

## Task 7: README.md

**Files:**
- Create: `README.md`

- [ ] **Step 1: Write README.md**

```markdown
# Multi-Agent Painter & Critic

A two-agent system built with the [AG2](https://docs.ag2.ai/) framework in which a Painter agent
draws a digital painting iteratively and a Critic agent evaluates each round using vision.

## Subject Prompt

> "a sunset over the ocean with orange sky and calm water"

(Configurable via `--subject` at runtime.)

## How to Run

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run (10 rounds, default subject)

```bash
python painter_critic.py --subject "a sunset over the ocean" --rounds 10
```

### Options

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--subject` | Yes | — | The drawing subject/prompt |
| `--rounds` | No | 10 | Number of Painter–Critic iterations |

### Output

All output is saved to the `output/` directory:
- `round_01.png` … `round_10.png` — canvas state after each Painter turn
- `conversation_log.txt` — full text conversation between agents

## Design Decisions

### Agent Pattern: Two-Agent Sequential Chat

The Critic calls `critic.initiate_chat(painter, max_turns=rounds*2)`. This means:
- The Critic sends the initial "start drawing" prompt (not counted in max_turns)
- Turns alternate: Painter draws → Critic critiques → Painter draws → ...
- With `max_turns = rounds * 2`, we get exactly `rounds` draws and `rounds` critiques
- `max_consecutive_auto_reply = rounds` is also set on the Critic as a secondary stop

This pattern is idiomatic AG2 and avoids manual loop orchestration.

### Why register_hook for Image Injection

Both agents need to see the current canvas. Rather than giving each agent a `get_canvas` tool
(which the LLM might forget to call), we use `register_hook("process_all_messages_before_reply", ...)`
on both agents. This hook runs automatically before every LLM call and prepends the current canvas
as a base64-encoded PNG in OpenAI vision format. The injection is transparent and reliable.

### Why register_reply for Round Saving

The Critic's `register_reply` hook (position=0, returns `(False, None)`) runs before the Critic's
LLM is called — i.e., after the Painter has finished drawing. This is the correct moment to save
the round's canvas as `round_NN.png`.

### Drawing Tools

Three pixel-level tools give the Painter a range from fine detail to large fills:

| Tool | Best for |
|------|---------|
| `draw_pixels` | Fine detail, gradients, precise color placement |
| `draw_line` | Outlines, rays, horizon lines, stems |
| `draw_filled_rectangle` | Sky, sea, ground, large background regions |

### Canvas

200×200 RGB PIL image. Tools mutate it in-place via module-level shared state.
Canvas is reset to white at the start of each `run()` call.

## Observations on Output Images

*(Fill this in after running the 10-round session.)*

- **What went well:** 
- **What went wrong:** 
- **Surprising behavior:** 
```

- [ ] **Step 2: Fill in observations after running the full session**

Open `README.md` and complete the "Observations" section based on actual round images.

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: add README with design decisions and run instructions"
```

---

## Self-Review Checklist

| Spec Requirement | Covered by |
|-----------------|-----------|
| AG2 ConversableAgent for both agents | Task 4 build_agents |
| 200×200 canvas | Task 1 CANVAS_SIZE=200 |
| ≥3 drawing tools | Task 2 (draw_pixels, draw_line, draw_filled_rectangle) |
| Configurable subject prompt | Task 5 --subject CLI arg |
| Both agents see canvas via vision | Task 3 inject_canvas_into_messages hook, registered on both agents |
| Configurable rounds (≥10) | Task 5 --rounds CLI arg, default 10 |
| Each round saves an image | Task 4 make_critic_round_hook → save_canvas |
| AG2 initiate_chat conversation mechanism | Task 5 critic.initiate_chat + max_turns + max_consecutive_auto_reply |
| README.md | Task 7 |
| round_01.png, round_05.png, round_10.png | Task 6 Step 5 |
| conversation_log.txt | Task 5 run() log writing |
