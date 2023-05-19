from typing import Tuple
import concurrent.futures
from random import uniform, randint, random, choice
import textwrap
from colorsys import hls_to_rgb
from bounding_box import SimpleBoundingBox

from fontTools.ttLib import TTFont
import numpy as np
import cv2 as cv
from PIL import Image, ImageFont, ImageDraw
from PIL.Image import Resampling


def lerp(a, b, fac):
    return a + (b - a) * fac


def pad_image(
    img: np.ndarray, padding: Tuple[int, int, int, int] = (0, 0, 0, 0), color: int = 255
):
    return np.pad(
        img, (padding[:2], padding[2:], (0, 0)), mode="constant", constant_values=color
    )


def wrap_text_to_match_aspect_ratio(text: str, font: ImageFont, aspect_ratio: float):
    # calculate line character width based on aspect ratio
    n_characters = len(text)

    text_bbox = font.getbbox(text)
    text_bbox = SimpleBoundingBox(text_bbox[:2], text_bbox[2:], np.int32)
    character_size = text_bbox.get_size()
    character_size[0] /= n_characters

    text_area = character_size[0] * character_size[1] * n_characters

    size = np.ceil(
        np.array(
            (text_area**0.5 * aspect_ratio, text_area**0.5 / aspect_ratio),
            np.float32,
        )
        / character_size
    ).astype(np.int32)

    # wrap text based on calculated width
    return textwrap.fill(
        text=text, width=size[0], break_long_words=False, break_on_hyphens=False
    )


def calculate_font_scale(text: str, font: ImageFont, resolution: int, draw: ImageDraw):
    # find font scale that roughly matches resolution
    text_bbox = draw.multiline_textbbox((0, 0), text, font=font)
    text_bbox = SimpleBoundingBox(text_bbox[:2], text_bbox[2:], np.int32)

    return resolution / np.amax(text_bbox.get_size())


def apply_colors(color0, color1, alpha):
    # to float
    alpha = alpha.astype(np.float32) / 255
    img = lerp(color0, color1, alpha)
    # to uint
    img = np.rint(img).astype(np.uint8)
    return cv.cvtColor(img, cv.COLOR_RGB2BGR)


def render_text_mask(
    text: str,
    font: ImageFont,
    resolution: int,
    text_aspect_ratio: float,
    alignment: str = "left",
) -> Tuple[str, np.ndarray]:
    text = wrap_text_to_match_aspect_ratio(text, font, text_aspect_ratio)

    # temp draw for getting font bbox
    draw = ImageDraw.Draw(Image.new("RGB", (0, 0)))
    font_scale = calculate_font_scale(text, font, resolution, draw)
    font_size = int(font.size * font_scale)
    # scale font
    font = font.font_variant(size=font_size)

    # calculate image resolution
    text_bbox = draw.multiline_textbbox((0, 0), text, font=font)
    text_bbox = SimpleBoundingBox(text_bbox[:2], text_bbox[2:], np.int32)
    # render text
    alpha = Image.new("RGB", text_bbox.get_size().tolist(), color=(255, 255, 255))
    draw = ImageDraw.Draw(alpha)
    draw.text(
        (-text_bbox[0][:]).tolist(), text, fill=(0, 0, 0), font=font, align=alignment
    )

    return text, alpha, font_size


def render_text(
    text: str,
    font: ImageFont,
    resolution: int,
    text_aspect_ratio: float,
    angle: int = 0,
    padding: Tuple[int, int, int, int] = (0, 0, 0, 0),
    alignment: str = "left",
    font_color: Tuple[int, int, int] = (0, 0, 0),
    background_color: Tuple[int, int, int] = (255, 255, 255),
) -> Tuple[str, np.ndarray, Tuple[Tuple[int, int], Tuple[int, int]]]:
    white = (255, 255, 255)

    text, alpha, font_size = render_text_mask(
        text, font, resolution, text_aspect_ratio, alignment
    )

    # rotate image
    alpha = alpha.rotate(
        angle, resample=Resampling.BICUBIC, expand=True, fillcolor=white
    )
    # to opencv
    alpha = np.array(alpha)
    # add padding
    alpha = pad_image(alpha, padding, white[0])
    # apply colors
    img = apply_colors(font_color, background_color, alpha)

    return text, img, alpha, font_size


def is_char_supported_by_font(font: TTFont, char: str) -> bool:
    return any(
        ord(char) in k.cmap.keys() for k in font["cmap"].tables if hasattr(k, "cmap")
    )


def is_text_supported_by_font(text: str, font: TTFont):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        return all(executor.map(lambda c: is_char_supported_by_font(font, c), text))


def hls_to_int_rgb(hue: float, lightness: float, saturation: float):
    col = hls_to_rgb(hue, lightness, saturation)
    return np.floor(np.array(col, np.float32) * 255).astype(np.int32)


def generate(text: str, font: ImageFont):
    hue, lightness, saturation = [random() for _ in range(3)]
    # make saturation curve steeper
    saturation = (saturation**2) * (3 - 2 * saturation)
    font_color = hls_to_int_rgb(hue, lightness * 0.6, saturation * 0.86)

    text_rotation_angle = randint(-45, 45)

    aspect_ratio = uniform(0.5, 2)
    padding = [randint(0, 64) for _ in range(4)]
    alignment = choice(["left", "center", "right"])

    return (
        *render_text(
            text,
            font,
            512,
            aspect_ratio,
            text_rotation_angle,
            padding,
            alignment,
            font_color,
        ),
        font_color,
        text_rotation_angle,
    )
