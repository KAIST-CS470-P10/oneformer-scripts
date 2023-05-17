"""

"""

from dataclasses import dataclass
from typing import Tuple

from transformers import OneFormerProcessor, OneFormerForUniversalSegmentation
from PIL import Image
import requests
import torch
import torchvision.utils as tvu
import tyro

@dataclass
class Args:

    oneformer_name: str = "shi-labs/oneformer_coco_swin_large"
    """Name of OneFormer pre-trained model. Set to `shi-labs/oneformer_coco_swin_large` by default."""

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
    processor, model = load_oneformer(args.oneformer_name)

    raise NotImplementedError("TODO: Implement this function")

if __name__ == "__main__":
    main(tyro.cli(Args))