# Multi-Agent Painter & Critic

A two-agent system built with the [AG2](https://docs.ag2.ai/) framework in which a Painter agent
draws a digital painting iteratively and a Critic agent evaluates each round using vision.

## Subject Prompt

> "a house with a door, a window and a chimney. The house is in a grass field and a tree is close to the house"

## How to Run

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run

```bash
python painter_critic.py --subject "a house with a door, a window and a chimney. The house is in a grass field and a tree is close to the house" --rounds 10
```

### Options

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--subject` | Yes | — | The drawing subject/prompt |
| `--rounds` | No | 10 | Maximum number of Painter–Critic iterations |

### Output

All output is saved to the `output/` directory:
- `round_01.png` … `round_NN.png` — canvas state after each Painter turn
- `conversation_log.txt` — full text conversation between agents (image data stripped)

The session may end before `--rounds` is reached if the Critic decides the drawing is complete (see Early Termination below).

**Note:** The `output/` directory is not cleared between runs. Remove or rename it before re-running to avoid mixing images from different sessions:
```bash
rm -rf output/
```

---

## Design Decisions

### Agent Pattern: Two-Agent Sequential Chat

The system uses AG2's two-agent chat pattern (`critic.initiate_chat(painter)`), where the **Critic is the initiator**. The conversation alternates:

1. Critic sends a drawing instruction (one specific element to draw)
2. Painter calls one drawing tool and replies with a brief description
3. Critic receives the canvas image (via hook), evaluates it, and sends the next instruction
4. Repeat until the drawing is complete or `--rounds` is reached

**Why the Critic initiates?**
The Critic is the outer loop agent — it controls pacing, saves the canvas at each round, and decides when to terminate. Having it initiate keeps all session-management logic on one side.

**Why one element per round?**
Early versions instructed the Painter to call 3–5 tools per message. This caused over-painting: the Painter would redraw established areas with new shapes, accumulating noise instead of improving. One tool call per round makes each change deliberate, traceable, and easy for the Critic to evaluate.

### Image Injection via `process_all_messages_before_reply` Hook

Both agents register a hook that injects the current canvas as a base64-encoded PNG into the last message before each reply:

```python
def inject_canvas_into_messages(messages: list[dict]) -> list[dict]:
    ...
```

This gives both agents visual context on every turn without any explicit tool call — the Painter sees the canvas before drawing, and the Critic sees it before critiquing.

**Why a hook instead of a `get_canvas` tool?**
A tool requires an extra round-trip (call → result → LLM reply). A hook is zero-overhead: the image is already in the message by the time the LLM is invoked. It also avoids the agent "forgetting" to call the tool.

**Important implementation note:** In AG2, `process_all_messages_before_reply` hooks run *before* the reply function list, including `check_termination_and_human_reply`. This means the hook converts the last message's content from a plain string to a multimodal list `[{image_url}, {text}]` before `is_termination_msg` is evaluated. The termination check must therefore handle both content formats:

```python
def _msg_is_done(msg: dict) -> bool:
    content = msg.get("content") or ""
    if isinstance(content, list):
        content = " ".join(p.get("text", "") for p in content if isinstance(p, dict))
    return content == "DONE" or "[done]" in content.lower()
```

### Round Saving via `register_reply` Hook

The Critic registers a reply handler at `position=0` that fires before its LLM reply each turn:

```python
make_critic_round_hook(num_rounds)
```

This hook:
- Skips tool_calls and tool_result messages (fires only on the Painter's text summary)
- Increments the round counter and saves `output/round_NN.png`
- Returns `(True, "DONE")` when the maximum round count is reached, terminating the session

### Early Termination by the Critic

The Critic is instructed to append `[DONE]` to its message when the drawing is complete — meaning all major parts of the subject are represented as clear, recognizable flat shapes. The Painter's `is_termination_msg` detects this and stops replying, ending the session before `--rounds` is reached.

This prevents unnecessary rounds after the drawing is finished and avoids the degradation that comes from the Critic inventing improvements on an already-complete image.

### AG2 Tool Execution Architecture: Why Tools Are Registered on Both Agents

In `critic.initiate_chat(painter)`, the Critic is the **outer loop agent**. When AG2 encounters Painter tool_calls in the message history, it is the Critic's `generate_tool_calls_reply` that executes them. Tools must therefore be registered on both:

```python
# Painter: LLM decides which tool to call + executes its own tools
register_function(draw_triangle, caller=painter, executor=painter, ...)

# Critic: executes Painter's tool_calls on the Critic's turn
critic.register_for_execution(name="draw_triangle")(draw_triangle)
```

Without registration on the Critic, tool execution fails with "not found" errors at runtime.

### Drawing Tools: Five Geometric Primitives

The system draws exclusively with flat, solid-color geometric shapes — no gradients, blending, or texture. This matches what an LLM can reliably express as discrete tool arguments.

| Tool | Signature | Use cases |
|------|-----------|-----------|
| `draw_filled_rectangle` | `x, y, width, height, r, g, b` | Walls, sky, backgrounds, doors, windows |
| `draw_circle` | `cx, cy, radius, r, g, b` | Sun, wheels, tree foliage, eyes |
| `draw_triangle` | `x1,y1, x2,y2, x3,y3, r, g, b` | Roofs, tree canopies, mountain peaks |
| `draw_polygon` | `points: list[{x,y}], r, g, b` | Irregular shapes with 4+ vertices |
| `draw_line` | `x1,y1, x2,y2, r, g, b, width` | Outlines, borders, trunks, horizon lines |

`draw_triangle` was added as a convenience wrapper around `draw_polygon` — it accepts 6 plain coordinates rather than a list of dicts, which is easier for the LLM to produce reliably.

### Critic Evaluation Style: Geometric / Child-Art Standard

The Critic evaluates drawings as flat geometric illustrations, not photorealistic art. A brown rectangle is a valid trunk. A green circle is a valid tree canopy. A dark red triangle is a valid roof.

This framing prevents the Critic from suggesting improvements that the available tools cannot achieve (smooth curves, gradients, shading), which was the main cause of drawing degradation in earlier versions.

The Critic uses a **PRESERVE / IMPROVE** feedback format each round:
- **PRESERVE** lists every element already drawn, with bounding boxes, so the Painter never repaints them
- **IMPROVE** contains exactly one instruction — the single most important missing element, with an exact tool call, coordinates, and RGB values

---

## Observations on Output Images

The example run drew a house scene over **8 rounds** before the Critic issued `[DONE]`.

### Round-by-Round Progression

| Round | What was added |
|-------|---------------|
| 1 | House body — brown rectangle (x 50–150, y 100–170) |
| 2 | Roof — dark brown triangle, apex (100,60), base (40–160, y 100) |
| 3 | Door — dark brown rectangle (x 90–110, y 140–170) |
| 4 | Window — light blue square (x 60–85, y 110–135) |
| 5 | Chimney — brown rectangle (x 135–145, y 70–90) |
| 6 | Tree trunk — green rectangle (x 20–45, y 110–150) |
| 7 | Tree foliage — green circle, center (33,90), radius 25 |
| 8 | Grass field — bright green rectangle (x 0–200, y 170–200) |

### What Went Well

- **One-element-per-round discipline worked**: Every round made exactly one visible change. The PRESERVE list grew correctly, and no previously drawn area was repainted.
- **Early termination worked**: The Critic correctly identified that all subject elements were present after round 8 and issued `[DONE]`, stopping 2 rounds early without any degradation.
- **Clear recognition**: The final image is immediately recognizable as the described scene — house with roof, chimney, door, window, a tree, and a grass field.
- **Critic instructions were specific**: Each IMPROVE item named an exact tool, coordinates, and RGB color, so the Painter never had to interpret vague feedback.
- **PRESERVE list prevented over-painting**: By explicitly listing bounding boxes of completed elements, the Critic stopped the Painter from re-drawing areas that were already correct.

### What Went Wrong or Was Unexpected

- **Tree trunk color**: The Critic suggested RGB (34, 139, 34) for the tree trunk — forest green instead of brown. A tree trunk rendered in the same green as the foliage reduces visual clarity. The Critic's color suggestions are sometimes semantically correct (green = nature) but visually suboptimal.

- **No sky drawn**: The white canvas background was left as the sky. The Critic never suggested filling the sky, focusing instead on the subject elements. A background fill in round 1 or 2 would improve the overall composition.

- **Chimney partially behind roof**: The chimney rectangle (x 135–145, y 70–90) overlaps the roof triangle. Since shapes are drawn in order with no z-ordering control, the chimney appears on top of the roof rather than growing out of it. Coordinate planning would require the LLM to reason about overlap.

### What Could Be Improved

1. **Sky and background**: Instruct the Painter to fill the background color in round 1, before drawing subject elements, to avoid the white-canvas-as-sky problem.

2. **Color guidance in Critic**: Add heuristics or examples to the Critic prompt for common colors (trunk = brown, sky = blue/orange, grass = green) to reduce semantically wrong color choices.

3. **Tool diversity**: A gradient fill or a `draw_ellipse` (non-circle) would improve expressiveness for subjects like hills, clouds, or oval windows.

4. **Dynamic round count**: Currently `--rounds` sets a hard maximum. A smarter default (e.g., `--rounds auto`) that runs until `[DONE]` with no upper bound other than a safety cap could simplify usage.
