# Handwritten OCR dataset rendering

Generate and render annotated images with text on a virtual paper with folds and other types of damage using PIL and Blender 3D.

## Installation

1. Clone the repository:

```
git clone --recursive https://github.com/GbotHQ/ocr-dataset-rendering.git
```

2. Install the required packages:

```
cd ocr-dataset-rendering
pip install -r requirements.txt
```

3. Download Blender:

```
cd src/Blender_3D_document_rendering_pipeline
bash download_blender_binary.sh
```

## Usage

example usage:
```
python main.py --n_samples 2 --blender_path "C:/Program Files/Blender Foundation/Blender 3.4/blender.exe" --output_dir "$output_dir" --device "cpu" --resolution_x 512 --resolution_y 512 --compression_level 9
```
```
python main.py --help
```
