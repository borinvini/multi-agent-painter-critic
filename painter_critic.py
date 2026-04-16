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
        # NOTE: 'canvas' is resolved from module scope at call time so that
        # reset() in tests and run() in production can rebind it effectively.
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


def make_critic_round_hook():
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
        # Only save and count when the last message is NOT a tool_calls message.
        # When the last message has tool_calls, the critic is about to execute them
        # (tools haven't run yet). When it's a regular text message, tools have
        # already been executed and critic is about to generate a critique.
        has_tool_calls = bool(last.get("tool_calls"))
        if not has_tool_calls:
            round_counter[0] += 1
            save_canvas(round_counter[0])
            print(f"\n[Round {round_counter[0]}] Canvas saved as output/round_{round_counter[0]:02d}.png")
        return False, None
    return hook


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
            }
        ],
        "cache_seed": None,
    }

    painter = ConversableAgent(
        name="Painter",
        system_message=(
            f"You are a Painter agent. Your job is to draw on a {CANVAS_SIZE}x{CANVAS_SIZE} pixel canvas "
            f"using your drawing tools.\nYou are drawing: {subject}\n\n"
            "IMPORTANT RULES:\n"
            "1. When you receive a drawing request or feedback, respond with MULTIPLE tool calls in a SINGLE "
            "   message — call draw_filled_rectangle, draw_line, and/or draw_pixels together in one turn. "
            "   Aim for at least 3-5 tool calls per message to make visible progress.\n"
            "2. After your tool calls, include a brief text description of what you drew.\n"
            "3. Do NOT call just one tool and wait — batch all your drawing for this round into one response.\n"
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

    # Register drawing tools for execution on critic too:
    # In the max_turns chat loop, when critic.generate_reply() is called and the last
    # message in history contains Painter's tool_calls, critic is the one that executes
    # them via generate_tool_calls_reply. Without this the tools return "not found".
    critic.register_for_execution(name="draw_pixels")(draw_pixels)
    critic.register_for_execution(name="draw_line")(draw_line)
    critic.register_for_execution(name="draw_filled_rectangle")(draw_filled_rectangle)

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
