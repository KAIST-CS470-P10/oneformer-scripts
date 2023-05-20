"""

"""

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, List

from transformers import OneFormerProcessor, OneFormerForUniversalSegmentation
from PIL import Image
import torch
import torchvision.utils as tvu
import tyro

@dataclass
class Args:

    image_dir: str
    """Directory containing input images."""
    oneformer_name: str = "shi-labs/oneformer_coco_swin_large"
    """Name of OneFormer pre-trained model. Set to `shi-labs/oneformer_coco_swin_large` by default."""

def find_images(directory: str) -> List[str]:
    """Looks for images in the given directory."""
    
    if not isinstance(directory, Path):
        directory = Path(directory)
    assert directory.exists(), f"Directory {directory} does not exist"

    # look for files with extension .jpg, .png, .jpeg
    image_files = directory.glob("*.jpg") + directory.glob("*.png") + directory.glob("*.jpeg")
    image_files = [str(f) for f in image_files]

    print(f"Found {len(image_files)} images in {directory}")

    return image_files


def load_oneformer(pretrained_model_name: str) -> Tuple[OneFormerProcessor, OneFormerForUniversalSegmentation]:
    """Loads OneFormer model and processor"""
    processor = OneFormerProcessor.from_pretrained(
        pretrained_model_name
    )
    model = OneFormerForUniversalSegmentation.from_pretrained(
        pretrained_model_name
    )
    return processor, model

def main(args: Args):
    """Main function"""
    
    files_to_process = find_images(args.image_dir)

    processor, model = load_oneformer(args.oneformer_name)

    raise NotImplementedError("TODO: Implement this function")

if __name__ == "__main__":
    main(tyro.cli(Args))