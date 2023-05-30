from random import seed

import numpy as np
import cv2 as cv
from PIL import ImageFont

from font_rendering import generate
from bounding_box_math import calculate_char_bboxes


class TestFontRenderer:
    def setup_class(self):
        self.font = ImageFont.truetype("tests/test_assets/SilentReaction.ttf", 42)
        self.input_text = "The quick brown fox"

        self.img = cv.imread("tests/test_assets/test_img_font_rendering_generate.png")
        self.alpha = cv.imread(
            "tests/test_assets/test_alpha_font_rendering_generate.png"
        )

    def generate_render_text_test_data(self):
        seed(0)

        (
            text,
            img,
            font_size,
            xy,
            line_offsets,
            padding,
            font_color,
            text_rotation_angle,
            resolution_before_rotation,
        ) = generate(self.input_text, self.font)

        compression = [cv.IMWRITE_PNG_COMPRESSION, 7]
        cv.imwrite(
            "tests/test_assets/test_img_font_rendering_generate.png", img, compression
        )
        test_dict = {
            "text": text,
            "font_size": font_size,
            "xy": xy,
            "line_offsets": line_offsets,
            "padding": padding,
            "font_color": font_color,
            "text_rotation_angle": text_rotation_angle,
            "resolution_before_rotation": resolution_before_rotation,
        }

        print(test_dict)
    
    def test_character_bounding_boxes(self):
        xy = [6, 21]
        bboxes = calculate_char_bboxes(xy, self.input_text, self.font)
        assert list(bboxes[0].keys()) == ["bbox", "mask", "char_index", "line_index"]

    def test_render_text(self):
        """Test if text rendering produces expected results"""
        seed(0)

        (
            text,
            img,
            font_size,
            xy,
            line_offsets,
            padding,
            font_color,
            text_rotation_angle,
            resolution_before_rotation,
        ) = generate(self.input_text, self.font, 512)

        np.testing.assert_array_equal(img, self.img)

        assert text == "The quick brown\nfox"
        assert font_size == 82
        assert xy == [6, 21]
        assert line_offsets == [0, 208]
        assert padding == [62, 51, 38, 61]
        assert font_color.tolist() == [154, 77, 148]
        assert text_rotation_angle == -12
        assert resolution_before_rotation == (508, 192)
