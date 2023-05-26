from random import seed

import numpy as np
import cv2 as cv
from PIL import ImageFont

from font_rendering import generate


class TestFontRenderer:
    def setup_class(self):
        self.font = ImageFont.truetype("tests/assets/SilentReaction.ttf", 42)
        self.input_text = "The quick brown fox"

        self.img = cv.imread("tests/assets/test_img_font_rendering_generate.png")
        self.alpha = cv.imread("tests/assets/test_alpha_font_rendering_generate.png")

    def imshow(self, img):
        cv.imshow("test", img)
        cv.waitKey(1000)
        cv.destroyAllWindows()

    def test_render_text(self):
        """Test if text rendering produces expected results"""
        seed(0)
        text, img, alpha, bbox, font_size, font_color, text_rotation_angle = generate(
            self.input_text, self.font
        )

        assert text == "The quick brown\nfox"

        np.testing.assert_array_equal(img, self.img)
        np.testing.assert_array_equal(alpha, self.alpha)

        np.testing.assert_array_equal(
            bbox.points,
            np.array(((63, 97), (251, 57), (355, 547), (167, 587)), np.int32)
        )
        assert font_size == 82
        assert font_color.tolist() == [154, 77, 148]
        assert text_rotation_angle == -12