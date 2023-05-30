from typing import Tuple, Union
from random import random, choice
from pathlib import Path as pth
from shutil import rmtree
from io import BytesIO
import threading
import itertools
import tempfile
import re
from functools import partial

from p_tqdm import t_map, p_umap
import numpy as np
import cv2 as cv
from fontTools.ttLib import TTFont
from PIL import ImageFont

from font_rendering import generate, is_text_supported_by_font
from Blender_3D_document_rendering_pipeline.src import config


class SampleInfo:
    def __init__(
        self,
        text: str,
        config: config.Config,
        anchor: Tuple[int, int],
        line_offsets,
        padding,
        image_path: pth,
        font_path: str,
        font_color: Tuple[int, int, int],
        font_size: int,
        text_rotation_angle: int,
        text_image_resolution: Tuple[int, int],
        output_image_resolution: Tuple[int, int],
        output_image_path: pth,
        output_coordinates_path: pth,
        compression_level: int,
        resolution_before_rotation: Tuple[int, int],
    ):
        self.text = text
        self.config = config
        self.anchor = anchor
        self.line_offsets = line_offsets
        self.padding = padding
        self.image_path = image_path
        self.font_path = font_path
        self.font_color = font_color
        self.font_size = font_size
        self.text_rotation_angle = text_rotation_angle
        self.text_image_resolution = text_image_resolution
        self.output_image_resolution = output_image_resolution
        self.output_image_path = output_image_path
        self.output_coordinates_path = output_coordinates_path
        self.compression_level = compression_level
        self.resolution_before_rotation = resolution_before_rotation


def mkdir(path: pth):
    if path.is_dir():
        rmtree(path)
    path.mkdir(parents=True)


def assign_material_to_conf(material, conf):
    material = {k: str(material[k].resolve()) for k in material}

    conf.ground.albedo_tex = material["albedo"]
    conf.ground.roughness_tex = material["roughness"]
    conf.ground.displacement_tex = material["displacement"]


def break_up_sample(text_sample):
    # make sure that punctuation has the correct spacing
    text_sample = re.sub(r"\s*([.,?!])\s*", r"\1 ", text_sample)

    text_sample = [k.strip() for k in text_sample.split(", ")]
    text_sample = [k for k in text_sample if k]

    # randomly merge back some adjecent strings
    i = 0
    while (i + 1) < len(text_sample):
        if random() < 0.15:
            text_sample[i] += f", {text_sample.pop(i + 1)}"
        i += 1

    return text_sample


def break_up_samples(text_samples):
    text_samples = map(break_up_sample, text_samples)
    return list(itertools.chain.from_iterable(text_samples))


def get_text_and_font(shuffled_text_dataset_iter, random_font_iter):
    while True:
        # find a font that supports all characters in text
        text = next(shuffled_text_dataset_iter)["sentences"]
        text = choice(break_up_sample(text))
        for _ in range(20):
            font_path = next(random_font_iter)
            with open(font_path, "rb") as f:
                font_file = f.read()
            # few fonts are be broken and will raise an exception
            try:
                if is_text_supported_by_font(text, TTFont(BytesIO(font_file))):
                    break
            except Exception as e:
                print(e)
                continue
        else:
            # give up and try a different piece of text
            print("No font found that supports all characters in text")
            continue
        break

    return text, font_path


def write_config(
    img: np.ndarray,
    device: str,
    root_dir: pth,
    output_dir: pth,
    hdri_path,
    material,
    output_image_resolution,
    compression_level: int,
    image_path,
    config_path: pth,
) -> config.Config:
    resolution = np.array(img.shape[:2], np.float32)
    paper_size = resolution[::-1] / np.mean(resolution) * 25

    conf = config.Config(device, project_root=root_dir)

    conf.hdri.texture_path = str(pth(hdri_path).resolve())

    assign_material_to_conf(material, conf)

    conf.render.output_dir = str(output_dir)
    conf.render.resolution = output_image_resolution
    conf.render.cycles_samples = 2
    conf.render.compression_ratio = round(compression_level / 9 * 100)
    conf.paper.document_image_path = str(image_path.resolve(True))
    conf.paper.size = paper_size.tolist()
    conf.ground.visible = random() < 0.6

    config.write_config(config_path, conf)

    return conf


def save_images(
    img: np.ndarray,
    image_dir_path: pth,
    sample_id: str,
    compression_level: int,
):
    image_path = image_dir_path / f"{sample_id}.png"
    cv.imwrite(str(image_path), img, [cv.IMWRITE_PNG_COMPRESSION, compression_level])

    return image_path


def generate_sample(
    index: int,
    text: str,
    font_path: str,
    material,
    hdri_path,
    root_dir,
    output_dir,
    image_dir,
    config_dir,
    text_render_resolution,
    output_image_resolution,
    device,
    compression_level,
) -> Union[SampleInfo, None]:
    try:
        with open(font_path, "rb") as f:
            font_file = f.read()
        font = ImageFont.truetype(BytesIO(font_file), 42)
    except Exception as e:
        print(f"Error while generating sample {index}: {e}")
        return

    # few fonts are be broken and will raise an exception
    try:
        (
            text,
            img,
            font_size,
            anchor,
            line_offsets,
            padding,
            font_color,
            text_rotation_angle,
            resolution_before_rotation,
        ) = generate(text, font, text_render_resolution)
    except Exception as e:
        print(f"Error while generating sample {index}: {e}")
        return

    sample_id = f"sample_{index:08d}"

    image_path = save_images(img, image_dir, sample_id, compression_level)

    config_path = config_dir / f"{sample_id}.json"

    conf = write_config(
        img,
        device,
        root_dir,
        output_dir,
        hdri_path,
        material,
        output_image_resolution,
        compression_level,
        image_path,
        config_path,
    )

    out_dir = output_dir / f"{sample_id}"
    out_image_path = out_dir / "image0001.png"
    out_coordinates_path = out_dir / "coordinates0001.png"

    resolution = img.shape[:2]

    return SampleInfo(
        text,
        conf,
        anchor,
        line_offsets,
        padding,
        image_path,
        font_path,
        font_color,
        font_size,
        text_rotation_angle,
        resolution,
        output_image_resolution,
        out_image_path,
        out_coordinates_path,
        compression_level,
        resolution_before_rotation,
    )


def generate_samples(
    n_samples: int,
    device: str,
    output_image_resolution: Tuple[int, int],
    compression_level: int,
    root_dir: pth,
    output_dir: pth,
    config_dir: pth,
    random_font_iter,
    random_hdri_iter,
    random_material_iter,
    shuffled_dataset_iter,
    multiprocessing: bool = True,
):
    root_dir = pth(root_dir)
    output_dir = pth(output_dir)
    config_dir = pth(config_dir)

    text_render_resolution = int(max(output_image_resolution))

    image_dir = pth(tempfile.mkdtemp())
    print(image_dir)

    mkdir(config_dir)

    texts = []
    font_paths = []
    materials = []
    hdris = []
    while len(texts) < n_samples:
        try:
            text, font_path = get_text_and_font(shuffled_dataset_iter, random_font_iter)

            # use textures from a single material
            # or combine textures from different materials for more diversity
            material = (
                next(random_material_iter)
                # if random() < 0.2
                # else {
                #     k: next(random_material_iter)[k]
                #     for k in ["albedo", "roughness", "displacement"]
                # }
            )

            hdri_path = next(random_hdri_iter)

            texts.append(text)
            font_paths.append(font_path)
            materials.append(material)
            hdris.append(hdri_path)
        except Exception as e:
            print(e)

    generate_sample_func = partial(
        generate_sample,
        root_dir=root_dir,
        output_dir=output_dir,
        image_dir=image_dir,
        config_dir=config_dir,
        text_render_resolution=text_render_resolution,
        output_image_resolution=output_image_resolution,
        device=device,
        compression_level=compression_level,
    )

    if multiprocessing:
        generated_samples = p_umap(
            generate_sample_func,
            range(len(texts)),
            texts,
            font_paths,
            materials,
            hdris,
            desc="Generating 2D samples"
        )
    else:
        generated_samples = t_map(
            generate_sample_func,
            range(len(texts)),
            texts,
            font_paths,
            materials,
            hdris,
            desc="Generating 2D samples"
        )

    # remove failed samples
    generated_samples = [k for k in generated_samples if k]

    return generated_samples
