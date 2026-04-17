import sys
import os
import argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import base64
import io as _io
from PIL import Image
import painter_critic as pc


def reset():
    pc.canvas = Image.new("RGB", (pc.CANVAS_SIZE, pc.CANVAS_SIZE), (255, 255, 255))
    pc.round_counter[0] = 0


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


class TestDrawCircle:
    def test_center_pixel_correct_color(self):
        reset()
        pc.draw_circle(100, 100, 10, 255, 0, 0)
        assert pc.canvas.getpixel((100, 100)) == (255, 0, 0)

    def test_clamps_near_edge_no_error(self):
        reset()
        # Center at (0,0) radius 20: bounding box clips to canvas — no crash
        pc.draw_circle(0, 0, 20, 0, 255, 0)
        # With clamped bounds (0,0) to (19,19), check a pixel that should be drawn
        assert pc.canvas.getpixel((10, 10)) == (0, 255, 0)

    def test_returns_confirmation_string(self):
        reset()
        result = pc.draw_circle(50, 50, 5, 0, 0, 255)
        assert isinstance(result, str) and "circle" in result.lower()


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


class TestDrawTriangle:
    def test_interior_pixel_correct_color(self):
        reset()
        # Flat-bottom triangle: apex (100,50), bottom-left (50,150), bottom-right (150,150)
        # Point (100,120) is well inside
        pc.draw_triangle(100, 50, 50, 150, 150, 150, 255, 0, 0)
        assert pc.canvas.getpixel((100, 120)) == (255, 0, 0)

    def test_clamps_out_of_bounds_no_error(self):
        reset()
        pc.draw_triangle(-10, -10, 250, -10, 100, 100, 0, 255, 0)
        assert pc.canvas.getpixel((100, 50)) == (0, 255, 0)

    def test_returns_confirmation_string(self):
        reset()
        result = pc.draw_triangle(10, 10, 50, 10, 30, 50, 0, 0, 255)
        assert isinstance(result, str) and "triangle" in result.lower()


class TestCLI:
    def test_subject_required(self):
        import subprocess, sys
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        result = subprocess.run(
            [sys.executable, os.path.join(project_root, "painter_critic.py")],
            capture_output=True, text=True,
            cwd=project_root,
        )
        assert result.returncode != 0
        assert "subject" in result.stderr.lower() or "required" in result.stderr.lower()

    def test_default_rounds_is_10(self):
        args = pc.build_parser().parse_args(["--subject", "a cat"])
        assert args.rounds == 10
        assert args.subject == "a cat"
