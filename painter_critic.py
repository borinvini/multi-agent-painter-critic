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
    return f"Drew line ({x1},{y1})→({x2},{y2})."


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
    Fires before each Critic LLM call: increments round counter, saves canvas.
    Returns (False, None) to pass control to the next reply handler (the LLM).
    """
    def hook(recipient: ConversableAgent, messages: list, sender: ConversableAgent, config: Any) -> tuple[bool, Any]:
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
