"""
test_oneformer.py

A script for testing OneFormer model.
"""

from dataclasses import dataclass
from typing import Tuple, Optional, List, Set, Dict

from transformers import OneFormerProcessor, OneFormerForUniversalSegmentation
from PIL import Image
import requests
import torch
import torch.nn as nn
from torchtyping import TensorType
import torchvision.utils as tvu
import tyro
import gzip
import io

from utils.oneformer_post_processor import post_process_panoptic_segmentation

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

    url = (
        "https://huggingface.co/datasets/hf-internal-testing/fixtures_ade20k/resolve/main/ADE_val_00000001.jpg"
    )
    image = Image.open(requests.get(url, stream=True).raw)

    inputs = processor(image, ["panoptic"], return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    panoptic_results = post_process_panoptic_segmentation(
        outputs, target_sizes=[image.size[::-1]]
    )
    
    # Gzip
    buf = io.BytesIO()
    torch.save(panoptic_results, buf)
    buf.seek(0)
    with gzip.open("../data/onef_outputs/oneformer.ptz", 'wb') as f:
        f.write(buf.read())
    
    print("Done")

if __name__ == "__main__":
    main(tyro.cli(Args))


