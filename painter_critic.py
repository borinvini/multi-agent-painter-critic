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
