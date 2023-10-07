def download_from_grdrive_or_mega(
    path: Union[str, pth],
    gdrive_file_id: str,
    mega_file_id_and_key: str,
    desc: Optional[str]=None,
) -> pth:
    path = pth(path)
    if desc is None:
        desc = str(path.stem)

    print(f"Downloading {desc}...")

    # download from google drive, if failed, download from mega
    try:
        download_path = gdrive_download_and_extract(path, gdrive_file_id)
        if not download_path.is_dir():
            raise DownloadError(f"{desc} from Google Drive")
    except Exception as e:
        print(e)
        print(f"Attempting to download {desc} from Mega.nz")

        download_path = mega_download_and_extract(path, mega_file_id_and_key)
        if not download_path.is_dir():
            raise DownloadError(f"{desc} from Mega.nz")

    return download_path


def download_and_prepare_data():
    download_path = pth("../assets/").resolve()
    if not download_path.is_dir():
        download_path.mkdir(parents=True, exist_ok=True)

    print("Preparing text dataset...")
    _, texts_iter = get_text_dataset()

    fonts_path = download_from_grdrive_or_mega(
        download_path / "fonts",
        "1-K8EE0QsXfxaAV-5uOE6lhTLGicRZbW2",
        "itg2TKoJ#UCOEQqX7pPwAf9pguWVUuksX7orWtzK5n4SdI6CQqGc"
    )

    hdris_path = download_from_grdrive_or_mega(
        download_path / "hdris",
        "1BNCTqw5fenCK-D48-a7VQ234Aq3k45hu",
        "D5YWQKgR#fFRHe-HpbCc7-yAm3h5zxbR905o1hkrxKJDvQOsGOKk"
    )

    materials_path = download_from_grdrive_or_mega(
        download_path / "materials",
        "1-5dz5DMce-braCrhVIsqB58PvcyB6qyy",
        "W0hGyCzB#-Gldvyt6uGt9D6iT8kDL-T4CKGNwIKD0Yc4jR2MAgxo"
    )

    if not fonts_path.is_dir():
        raise DownloadError("materials")

    fonts_iter = get_fonts(fonts_path)
    hdris_iter = get_hdris(hdris_path)
    materials_iter = get_materials(materials_path)

    return texts_iter, fonts_iter, hdris_iter, materials_iter

def main(
    n_samples: int,
    blender_path: str,
    output_dir: str,
    device: str,
    resolution_x: int = 512,
    resolution_y: int = 512,
    compression_level: int = 9,
):
    """
    Runs the main pipeline for rendering handwritten text on a virtual piece of paper using Blender.
    It downloads and prepares data for rendering 3D documents using Blender.
    It generates samples, renders them using Blender, and post-processes them.
    The generated samples are saved in the specified output directory.

    Args:
        n_samples (int): The number of sample images to generate.
        blender_path (str): The path to the Blender executable.
        output_dir (str): The path to the directory where the rendered images will be saved.
        device (str): The device to use for rendering ('cpu', 'cuda' or 'optix').
        output_image_resolution (Tuple[int, int], optional): The resolution of the output images. Defaults to (512, 512).
        compression_level (int, optional): The png compression level to use when saving the output images.
                                           Must be between 0 and 9, with 0 being no compression and 9 being maximum compression.
                                           Defaults to 9.

    Raises:
        ValueError: If the output directory is not empty.

    Returns:
        None
    """

    device = device.upper()

    blender_path: pth = pth(blender_path)
    output_dir: pth = pth(output_dir)

    if device not in ["CPU", "CUDA", "OPTIX"]:
        raise ValueError(f"Invalid device: {device}")

    if not blender_path.is_file():
        raise ValueError(f"Blender path {blender_path} is not a valid file")

    if not output_dir.is_dir():
        output_dir.mkdir(parents=True, exist_ok=True)

    blender_path = blender_path.resolve()
    output_dir = output_dir.resolve()

    # Check if the output directory is empty
    if output_dir.is_dir() and list(output_dir.iterdir()):
        raise ValueError(f"Output directory {output_dir} is not empty")

    resolution = resolution_x, resolution_y

    texts, fonts, hdris, materials = download_and_prepare_data()

    root_dir = pth.cwd() / "Blender_3D_document_rendering_pipeline"

    temp_dir = pth(tempfile.mkdtemp())
    print(f"Saving temporary files to: {temp_dir}")

    config_dir = temp_dir / "configs"
    image_dir = temp_dir / "images"
    config_dir.mkdir()
    image_dir.mkdir()

    print("Generating samples...")
    generated_samples = generate_samples(
        n_samples=n_samples,
        device=device,
        output_image_resolution=resolution,
        compression_level=compression_level,
        root_dir=root_dir,
        output_dir=output_dir,
        config_dir=config_dir,
        image_dir=image_dir,
        random_font_iter=fonts,
        random_hdri_iter=hdris,
        random_material_iter=materials,
        shuffled_dataset_iter=texts,
    )

    print("Rendering samples using Blender...")
    output_dir.mkdir(parents=True, exist_ok=True)
    run_blender_command(blender_path, config_dir, output_dir, device)

    postprocess_samples(generated_samples)

    for k in generated_samples:
        output_dict = {
            "text": k.text,
            "bboxes": k.bounding_boxes,
            "font_path": str(k.font_path),
            "font_color": k.font_color.tolist(),
            "text_rotation_angle": k.text_rotation_angle,
            "resolution": k.output_image_resolution,
        }

        if k.bounding_boxes is None:
            print(f"Failed to calculate bounding boxes for {k.text}! Skipping")
            print(json.dumps(output_dict, indent=4))
            sample_output_dir = k.output_image_path.parent
            rmtree(sample_output_dir)
            continue

        with open(k.output_image_path.with_suffix(".json"), "w") as f:
            json.dump(output_dict, f, indent=4)

    print("Cleaning up temporary files...")
    rmtree(temp_dir)
