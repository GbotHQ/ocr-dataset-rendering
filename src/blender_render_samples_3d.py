from typing import Union
import subprocess
from pathlib import Path as pth
from shutil import rmtree


def mkdir(path: pth):
    if path.is_dir():
        rmtree(path)
    path.mkdir(parents=True)


def run_blender_command(
    blender_binary_path: Union[str, pth],
    config_dir: Union[str, pth],
    output_dir: Union[str, pth],
    device: str,
):
    if not pth(output_dir).is_dir():
        raise FileNotFoundError(
            f"Output directory {output_dir} does not exist, please create it first"
        )

    out = subprocess.run(
        [
            str(blender_binary_path),
            "./Blender_3D_document_rendering_pipeline/blender/scene.blend",
            "--background",
            "--factory-startup",
            "--threads",
            "0",
            "--engine",
            "CYCLES",
            "--enable-autoexec",
            "--python",
            "./Blender_3D_document_rendering_pipeline/src/main.py",
            "--",
            "--cycles-device",
            device,
            "--config_path",
            str(config_dir),
        ],
        capture_output=True,
    )
    print(out.stdout.decode("utf-8"))
    if out.returncode != 0:
        raise Exception(out.stderr.decode("utf-8"))
