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
