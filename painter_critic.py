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
    return f"Drew line ({x1},{y1})->({x2},{y2})."


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
    return f"Drew rectangle ({x0},{y0}) {width}×{height}."


def draw_circle(
    cx: Annotated[int, "Center x (0-199)"],
    cy: Annotated[int, "Center y (0-199)"],
    radius: Annotated[int, "Radius in pixels (1-199)"],
    r: Annotated[int, "Red 0-255"],
    g: Annotated[int, "Green 0-255"],
    b: Annotated[int, "Blue 0-255"],
) -> str:
    """Draw a filled circle on the canvas."""
    draw = ImageDraw.Draw(canvas)
    cx = max(0, min(CANVAS_SIZE - 1, cx))
    cy = max(0, min(CANVAS_SIZE - 1, cy))
    radius = max(1, radius)
    x0 = max(0, cx - radius)
    y0 = max(0, cy - radius)
    x1 = min(CANVAS_SIZE - 1, cx + radius)
    y1 = min(CANVAS_SIZE - 1, cy + radius)
    draw.ellipse(
        [(x0, y0), (x1, y1)],
        fill=(max(0, min(255, r)), max(0, min(255, g)), max(0, min(255, b))),
    )
    return f"Drew circle at ({cx},{cy}) radius {radius}."


def draw_polygon(
    points: Annotated[list[dict], "List of dicts with keys x (0-199) and y (0-199) defining polygon vertices"],
    r: Annotated[int, "Red 0-255"],
    g: Annotated[int, "Green 0-255"],
    b: Annotated[int, "Blue 0-255"],
) -> str:
    """Draw a filled polygon on the canvas."""
    if len(points) < 3:
        return f"draw_polygon requires at least 3 vertices, got {len(points)}."
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


def draw_triangle(
    x1: Annotated[int, "Vertex 1 x (0-199)"],
    y1: Annotated[int, "Vertex 1 y (0-199)"],
    x2: Annotated[int, "Vertex 2 x (0-199)"],
    y2: Annotated[int, "Vertex 2 y (0-199)"],
    x3: Annotated[int, "Vertex 3 x (0-199)"],
    y3: Annotated[int, "Vertex 3 y (0-199)"],
    r: Annotated[int, "Red 0-255"],
    g: Annotated[int, "Green 0-255"],
    b: Annotated[int, "Blue 0-255"],
) -> str:
    """Draw a filled triangle on the canvas."""
    draw = ImageDraw.Draw(canvas)
    coords = [
        (max(0, min(CANVAS_SIZE - 1, x1)), max(0, min(CANVAS_SIZE - 1, y1))),
        (max(0, min(CANVAS_SIZE - 1, x2)), max(0, min(CANVAS_SIZE - 1, y2))),
        (max(0, min(CANVAS_SIZE - 1, x3)), max(0, min(CANVAS_SIZE - 1, y3))),
    ]
    draw.polygon(coords, fill=(max(0, min(255, r)), max(0, min(255, g)), max(0, min(255, b))))
    return f"Drew triangle at ({x1},{y1}), ({x2},{y2}), ({x3},{y3})."


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
    # Only inject into plain-text messages; skip if content is already a list
    # (e.g., message was already processed by this hook in a prior call).
    if isinstance(text, str):
        last["content"] = [
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{canvas_to_base64()}"},
            },
            {"type": "text", "text": text},
        ]
    return messages[:-1] + [last]


def make_critic_round_hook(num_rounds: int):
    """
    Returns a register_reply function for the Critic.
    Fires before each Critic reply. Only saves canvas when the last message is NOT
    a tool_calls message (i.e., tools have already been executed and Critic is about
    to generate an LLM critique). Returns (False, None) to pass control to the next
    reply handler (the LLM).
    """
    def hook(recipient: ConversableAgent, messages: list, sender: ConversableAgent, config: Any) -> tuple[bool, Any]:
        if not messages:
            return False, None
        last = messages[-1]
        # Save only when the last message is a plain Painter text summary
        # (not a tool_calls request and not a tool result message).
        # This ensures one save per Painter drawing turn, capped at num_rounds.
        has_tool_calls = bool(last.get("tool_calls"))
        is_tool_result = last.get("role") == "tool"
        if not has_tool_calls and not is_tool_result and round_counter[0] < num_rounds:
            round_counter[0] += 1
            save_canvas(round_counter[0])
            print(f"\n[Round {round_counter[0]}] Canvas saved as output/round_{round_counter[0]:02d}.png")
            if round_counter[0] >= num_rounds:
                # All rounds complete — terminate the conversation immediately
                # instead of letting max_turns keep the chat running.
                return True, "DONE"
        return False, None
    return hook


def _msg_is_done(msg: dict) -> bool:
    """Return True if the message signals end-of-session ("DONE" or "[DONE]").

    process_all_messages_before_reply runs before check_termination_and_human_reply,
    so by the time is_termination_msg is called the inject_canvas hook has already
    converted the last message content from a plain string to a multimodal list
    [{image_url}, {text}].  We must handle both forms.
    """
    content = msg.get("content") or ""
    if isinstance(content, list):
        content = " ".join(p.get("text", "") for p in content if isinstance(p, dict))
    return content == "DONE" or "[done]" in content.lower()


def build_agents(subject: str, num_rounds: int) -> tuple[ConversableAgent, ConversableAgent]:
    """Create and configure the Painter and Critic agents.

    NOTE: This function does not reset round_counter or the canvas.
    Call run() (which resets both) instead of calling build_agents directly,
    or reset manually before calling if using in tests.
    """
    llm_config = {
        "config_list": [
            {
                "model": "openai/gpt-4.1-mini",
                "base_url": "https://5f5832nb90.execute-api.eu-central-1.amazonaws.com/v1",
                "api_key": "none",
                "price": [0.0004, 0.0016],
            }
        ],
        "cache_seed": None,
    }

    painter = ConversableAgent(
        name="Painter",
        system_message=(
            f"You are a Painter agent. Your job is to draw on a {CANVAS_SIZE}x{CANVAS_SIZE} pixel canvas "
            f"using your drawing tools.\nYou are drawing: {subject}\n\n"
            "DRAWING STYLE: This is a flat, geometric illustration — like a child's drawing. "
            "Use solid-colored rectangles, circles, and polygons to represent each part of the subject. "
            "Do not attempt gradients, shading, or texture. "
            "A flat brown rectangle is a perfect trunk. A green circle is a perfect tree canopy. "
            "A dark red triangle polygon is a perfect roof.\n\n"
            "TOOLS AVAILABLE:\n"
            "  draw_filled_rectangle — fill a rectangular area with a solid color\n"
            "  draw_circle           — draw a filled circle\n"
            "  draw_triangle         — draw a filled triangle (3 vertices as x1,y1,x2,y2,x3,y3)\n"
            "  draw_polygon          — draw a filled polygon (4+ vertices as a list of {x,y} dicts)\n"
            "  draw_line             — draw a straight line between two points (supports width)\n\n"
            "APPROACH:\n"
            "  Each round you draw exactly ONE element — whatever the Critic asks for, or the most\n"
            "  important missing part if it is the first round. Use a single tool call per round.\n"
            "  Do not repaint an area that already looks correct.\n\n"
            "NON-DESTRUCTIVE RULE:\n"
            "  Never use a filled shape to cover an area that already has non-white pixels unless "
            "explicitly correcting a mistake identified by the Critic.\n\n"
            "CALL RULE:\n"
            "  Make exactly 1 tool call per message, then briefly state what you drew."
        ),
        llm_config=llm_config,
        human_input_mode="NEVER",
        is_termination_msg=_msg_is_done,
    )

    critic = ConversableAgent(
        name="Critic",
        system_message=(
            f"You are an art Critic for a simple geometric drawing system.\n"
            f"The subject being drawn is: {subject}\n"
            f"Maximum rounds: {num_rounds}\n\n"
            "DRAWING STYLE — CRITICAL: This is a FLAT, GEOMETRIC illustration, like a child's drawing.\n"
            "The only available tools are: filled rectangles, filled circles, filled triangles, filled polygons, and lines.\n"
            "There are NO gradients, NO blending, NO smooth curves, NO texture, NO shadows.\n"
            "Evaluate the drawing ONLY by this standard: does it clearly show recognizable shapes\n"
            "that represent the subject? A flat rectangle is a perfect door. A triangle is a perfect\n"
            "roof. That is success — not photorealism.\n\n"
            "FEEDBACK FORMAT — use this exact structure every round:\n\n"
            "  PRESERVE (do not touch):\n"
            "  - [Shape name + bounding box, e.g. 'Roof triangle: x 40-160, y 40-90']\n\n"
            "  IMPROVE (exactly 1 item):\n"
            "  1. [Exact instruction: tool name, coordinates, RGB color, what it represents]\n\n"
            "RULES:\n"
            "  - Never ask the Painter questions. Give instructions only.\n"
            "  - Never suggest changes to PRESERVE zones.\n"
            "  - Exactly 1 item in IMPROVE per round — the single highest-impact missing element.\n"
            "  - If an improvement from the previous round was completed well, move it to PRESERVE.\n"
            "  - Every IMPROVE item must name an exact tool with pixel coordinates and RGB values.\n"
            "  - NEVER suggest: smooth edges, gradients, blending, texture, shading, lighting,\n"
            "    shadows, grass, floor, sky, or any background context elements.\n"
            "  - NEVER suggest re-drawing or refining an area that already has a recognizable shape.\n"
            "    Stacking more shapes on top of an existing shape creates noise, not improvement.\n\n"
            "EARLY TERMINATION:\n"
            "  After each round, ask yourself: does the drawing already show all major parts of the\n"
            "  subject as clear, recognizable flat shapes with nothing meaningful left to add?\n"
            "  If yes — write your final PRESERVE list (everything looks good) and end your message\n"
            "  with exactly: [DONE]\n"
            "  This stops the session immediately. Use it as soon as the drawing is complete.\n"
            "  Do NOT wait until the last round if the drawing is already finished."
        ),
        llm_config=llm_config,
        human_input_mode="NEVER",
    )

    # Register drawing tools: Painter is both the caller (LLM) and executor
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
    register_function(
        draw_polygon,
        caller=painter,
        executor=painter,
        name="draw_polygon",
        description=(
            "Draw a filled polygon on the canvas. "
            "Parameters: points (list of dicts with keys x (int 0-199) and y (int 0-199) "
            "defining the polygon vertices), r (int 0-255), g (int 0-255), b (int 0-255). "
            "Use for irregular shapes with 4+ vertices."
        ),
    )
    register_function(
        draw_triangle,
        caller=painter,
        executor=painter,
        name="draw_triangle",
        description=(
            "Draw a filled triangle on the canvas. "
            "Parameters: x1,y1 (int 0-199, vertex 1), x2,y2 (int 0-199, vertex 2), "
            "x3,y3 (int 0-199, vertex 3), r (int 0-255), g (int 0-255), b (int 0-255). "
            "Use for roofs, tree canopies, mountain peaks, arrow heads."
        ),
    )

    # Register drawing tools for execution on critic too:
    # In the max_turns chat loop, when critic.generate_reply() is called and the last
    # message in history contains Painter's tool_calls, critic is the one that executes
    # them via generate_tool_calls_reply. Without this the tools return "not found".
    critic.register_for_execution(name="draw_line")(draw_line)
    critic.register_for_execution(name="draw_filled_rectangle")(draw_filled_rectangle)
    critic.register_for_execution(name="draw_circle")(draw_circle)
    critic.register_for_execution(name="draw_polygon")(draw_polygon)
    critic.register_for_execution(name="draw_triangle")(draw_triangle)

    # Critic: save canvas + inject canvas image before each critique
    critic.register_reply(
        [ConversableAgent],
        make_critic_round_hook(num_rounds),
        position=0,
    )
    critic.register_hook("process_all_messages_before_reply", inject_canvas_into_messages)

    # Painter: inject current canvas image before each drawing turn
    painter.register_hook("process_all_messages_before_reply", inject_canvas_into_messages)

    return painter, critic


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

    # Each round requires ~4 message turns: Painter draws (tool_calls) →
    # Critic executes tools (tool_results) → Painter text → Critic critique.
    # Use num_rounds * 4 + 4 to allow full conversation flow.
    chat_result = critic.initiate_chat(
        painter,
        message=initial_message,
        max_turns=num_rounds * 4 + 4,
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


def build_parser() -> argparse.ArgumentParser:
    """Build and return the CLI argument parser."""
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
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run(args.subject, args.rounds)


if __name__ == "__main__":
    main()
