# Multi-Agent Painter & Critic

A two-agent system built with the [AG2](https://docs.ag2.ai/) framework in which a Painter agent
draws a digital painting iteratively and a Critic agent evaluates each round using vision.

## Subject Prompt

> "a sunset over the ocean with orange sky and calm water"

## How to Run

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run

```bash
python painter_critic.py --subject "a sunset over the ocean with orange sky and calm water" --rounds 10
```

### Options

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--subject` | Yes | — | The drawing subject/prompt |
| `--rounds` | No | 10 | Number of Painter–Critic iterations |

### Output

All output is saved to the `output/` directory:
- `round_01.png` … `round_10.png` — canvas state after each Painter turn  
- `conversation_log.txt` — full text conversation between agents (image data stripped)

## Design Decisions

### Agent Pattern: Sequential Chat Loop

The system uses a two-agent chat pattern where the **Critic initiates the conversation**:
1. Critic sends the initial drawing request to the Painter
2. Painter draws (calls tools) and provides a brief description
3. Critic evaluates the canvas image (injected via hook) and provides structured feedback
4. Loop repeats for `num_rounds` iterations

The loop is controlled via `max_turns=num_rounds * 4 + 4`. Each round requires approximately 4 message turns: Painter draws (tool_calls) → Critic executes tools (tool_results) → Painter text summary → Critic critique. This sizing ensures we capture the full conversation flow without exceeding the desired number of painting rounds.

The Critic initiates (rather than the Painter) because it simplifies state management—the Critic can be the outer loop agent that controls the conversation flow and saves state after each round.

### Image Injection via Hook (Multimodal Vision)

Both agents register a `process_all_messages_before_reply` hook that injects the current canvas as a base64-encoded PNG image into the last message before the agent replies:

```python
inject_canvas_into_messages(messages: list[dict]) -> list[dict]
```

This hook:
- Converts the canvas to base64 PNG
- Prepends an image_url message content block to the text
- Preserves the original text feedback/request below the image

**Why this approach over a `get_canvas` tool?**
- **Reliability**: Vision models work best when images are injected directly into the message history, allowing the model to correlate feedback and the current state in a single reply
- **Simplicity**: No need for tool execution overhead; the image is always available at reply time
- **Natural workflow**: The Critic can see what it's critiquing in real time, and the Painter can see feedback alongside the current state

### Round Saving via Register-Reply Hook

The Critic registers a reply handler (via `register_reply` with `position=0`) that fires **before** each Critic reply:

```python
make_critic_round_hook(num_rounds)
```

This hook:
- Checks if the last message is a plain Painter text summary (not a tool_calls or tool_result message)
- Only then increments the round counter and saves the canvas
- Returns `(False, None)` to pass control to the next reply handler (the LLM)

**Why `position=0` and before the LLM reply?**
- Saving at position=0 ensures the canvas is captured after the Painter has finished drawing (tool_calls have been executed)
- This guarantees one canvas save per Painter drawing turn, capped at `num_rounds`
- Firing before the LLM critique gives us a clean snapshot of the Painter's work before the Critic comments

### AG2 Tool Execution Architecture: Why Tools Are Registered on Both Agents

In `critic.initiate_chat(painter)`, the Critic is the **outer loop agent**. When AG2 processes messages:

1. The Critic generates a reply and sees Painter tool_calls in the message history
2. The Critic's `generate_tool_calls_reply` is called to execute those tools
3. Tools must be registered on the **executor** agent (the one calling `generate_tool_calls_reply`)

Therefore, drawing tools are registered twice:

```python
# On Painter (LLM caller + executor during Painter's turn)
register_function(draw_pixels, caller=painter, executor=painter, ...)

# On Critic (executor when Critic processes Painter's tool_calls in the loop)
critic.register_for_execution(name="draw_pixels")(draw_pixels)
```

This is a key architectural detail: **the outer loop agent (Critic) executes the inner loop agent's (Painter's) tools**. Without registering on the Critic, tool execution fails with "not found" errors.

### Drawing Tools: Three Complementary Operations

The system provides three drawing tools, each optimized for different aspects of the painting:

1. **`draw_filled_rectangle(x, y, width, height, r, g, b)`**
   - Fills large rectangular regions with solid color
   - Used for broad strokes: sky, ocean water, large color blocks
   - Efficient for covering large areas quickly

2. **`draw_line(x1, y1, x2, y2, r, g, b, width=2)`**
   - Draws straight lines with variable width
   - Used for horizons, water ripples, reflections, architectural elements
   - Supports width parameter for brush-like effects

3. **`draw_pixels(pixels: list[dict])`**
   - Batch-draws individual pixels with fine control
   - Used for fine detail: sun glow, subtle gradients, texture, small highlights
   - Requires at least 20 pixels per call to ensure visible progress

The Painter is instructed to batch multiple tool calls in a single message (3–5 calls recommended) to make visible progress in each turn.

## Observations on Output Images

### Initial State (Round 1)

Round 1 shows minimal progress—the canvas is still mostly white. The Painter attempts to draw the basic composition but encounters some tool execution issues (visible in the conversation log). This is expected behavior at the start of the iterative process.

### Mid-Progress (Round 5)

By Round 5, the composition has taken clear shape:
- **Sky**: Solid orange filling the top ~100 pixels, representing the warm sunset glow
- **Ocean**: Divided into two distinct sections—a darker blue horizon area and lighter blue water below
- **Horizon**: Horizontal lines and color transitions marking the sky-water boundary
- **Sun element**: A small bright yellow/orange shape near the horizon (initially rectangular, then refined to a rounder form)

The canvas shows approximately 40,000 non-white pixels (fully painted), confirming that major color blocks have been established.

### Final State (Round 10)

Round 10 represents the refinement phase:
- **Sky gradient**: More nuanced orange tones, with subtle variations suggesting depth
- **Sun**: Better-formed circular shape with smoother edges (achieved via pixel-level drawing)
- **Water reflections**: Added white and light orange pixels across the water to simulate sun reflections and shimmer
- **Wave details**: Additional lines and subtle texture suggesting calm but not static water
- **Overall cohesion**: The scene is more unified—the reflected light connects the sun to the water, enhancing the sunset atmosphere

The canvas remains at approximately 40,000 non-white pixels, confirming the Painter focused on refinement and quality over coverage in later rounds.

### What Went Well

- **Clear compositional structure**: The sky-ocean division was established by Round 1 and remained stable
- **Effective color palette**: Orange and blue create strong visual contrast appropriate to sunset
- **Iterative refinement**: Each round saw improvements in detail and naturalness (e.g., round sun instead of blocky shape, smoother transitions)
- **Agent feedback loop**: The Critic's comments were specific and actionable, guiding the Painter's next steps
- **Tool batching**: The Painter learned to call multiple tools per message, accelerating progress

### What Went Wrong or Was Unexpected

- **Round count overshoot**: `--rounds 10` produces 16 saved images (rounds 1–16) instead of exactly 10
  - This occurs because `max_turns=num_rounds * 4 + 4` allows extra conversation turns
  - The loop saves a canvas whenever the Critic is about to reply with a critique, which can happen more frequently than expected due to message structure
  - The Critic's round-saving logic caps at `num_rounds`, but earlier rounds may trigger multiple saves during tool error recovery

- **Tool validation errors**: Round 2–3 shows a Pydantic validation error when the Painter attempted to call `draw_pixels` with missing field `s` (likely a serialization issue)
  - This was recovered gracefully and didn't break the conversation
  - Subsequent calls succeeded, suggesting the agent self-corrected

- **Minimal final changes**: Rounds 11–16 show very small file sizes (592–932 bytes vs. 1700+ in earlier rounds)
  - This indicates the agents reached a stopping point or were making negligible updates
  - The Critic may have been satisfied with the image, leading to brief replies

### What Could Be Improved

1. **Round count accuracy**: Adjust the round-saving logic or `max_turns` calculation to guarantee exactly N saved rounds
   - Consider tracking rounds by Painter drawing turns (not Critic replies) for tighter control

2. **Tool error handling**: Add richer error messages and recovery hints to the Painter's system prompt
   - Teach it to validate tool parameters before calling

3. **Canvas state visibility**: Provide the Painter with explicit feedback on canvas coverage or pixel count
   - E.g., "Canvas is 30% painted; focus on the remaining areas"
   - This could guide the agent to more balanced compositions

4. **Termination conditions**: Implement a score-based stopping criterion
   - E.g., stop when the Critic's feedback becomes repetitive or when pixel changes fall below a threshold
   - Currently the loop runs for a fixed number of turns; dynamic stopping could be more efficient

5. **Multimodal prompting**: Guide the Painter with more specific visual targets
   - E.g., "Make the sun appear as a smooth circle with a warm glow, not a blocky shape"
   - Vision-language models respond better to visual descriptions than abstract goals

6. **Tool diversity**: Add additional tools for advanced effects
   - Gradient fills (linear or radial) for smoother color transitions
   - Ellipse/circle drawing for round shapes
   - Text rendering (if appropriate for the subject)
