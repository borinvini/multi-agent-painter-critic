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
