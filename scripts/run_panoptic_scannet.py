"""
run_panoptic_scannet.py

A script for running OneFormer on ScanNet images.
"""

from easydict import EasyDict
from functools import partial
from typeguard import typechecked
from typing import Dict, Type
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, List

import gzip
import numpy as np
from transformers import OneFormerProcessor, OneFormerForUniversalSegmentation
from PIL import Image
import torch
import torch.nn.functional as F
import torch_scatter
from torchtyping import TensorType, patch_typeguard
from tqdm import tqdm
from typeguard import typechecked
import tyro

try:
    from scripts.utils.constants import SCANNET_COLORS
    from scripts.utils.label_translator import COCOtoScanNet
    from scripts.utils.visualizer import Visualizer
except ImportError:
    print(
        "Import error occured. Possibly PYTHONPATH is not set. Please set PYTHONPATH to root of this repository."
    )
    exit(-1)

patch_typeguard()

@dataclass
class Args:

    image_dir: str
    """Directory containing input images."""
    tag: str
    """A tag used to label the processed data."""
    oneformer_name: str = "shi-labs/oneformer_coco_swin_large"
    """Name of OneFormer pre-trained model. Set to `shi-labs/oneformer_coco_swin_large` by default."""
    is_reduced_scannet: bool = True
    """A flag indicating whether to use reduced ScanNet classes. Set to True by default."""
    out_dir: str = "outputs"
    """Directory to save output images."""

def find_images(directory: str) -> List[str]:
    """Looks for images in the given directory."""
    
    if not isinstance(directory, Path):
        directory = Path(directory)
    assert directory.exists(), f"Directory {directory} does not exist"

    # look for files with extension .jpg, .png, .jpeg
    image_files = list(directory.glob("*.jpg")) + list(directory.glob("*.png")) + list(directory.glob("*.jpeg"))
    image_files = [str(f) for f in image_files]

    print(f"Found {len(image_files)} images in {directory}")

    return sorted(image_files)

def hex_to_rgb(x):
    return [int(x[i:i + 2], 16) for i in (1, 3, 5)]

def main(args: Args):
    """Main function"""

    # identify files to process    
    files_to_process = find_images(args.image_dir)

    # create output directories
    out_dir = Path(args.out_dir) / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Created directory: {str(out_dir)}")
    
    panoptic_dir = out_dir / "panoptic"
    panoptic_dir.mkdir(parents=True, exist_ok=True)
    print(f"Created directory: {str(panoptic_dir)}")

    vis_dir = out_dir / "visualization"
    vis_dir.mkdir(parents=True, exist_ok=True)
    print(f"Created directory: {str(vis_dir)}")

    # load OneFormer model
    processor, model = load_oneformer(args.oneformer_name)

    # initialize label translator
    coco_to_scannet = COCOtoScanNet(args.is_reduced_scannet)

    # initialize visualizer function
    data_tag = "reduced" if args.is_reduced_scannet else "extended"
    scannet_label_colors = [hex_to_rgb(color) for color in SCANNET_COLORS[data_tag]]
    
    visualizer_metadata = EasyDict(
        {
            "stuff_colors": scannet_label_colors,
            "thing_colors": scannet_label_colors,
            "stuff_classes": coco_to_scannet.scannet_class_names,
            "thing_classes": coco_to_scannet.scannet_class_names,
        },
    )
    visualizer_func = partial(
        Visualizer,
        metadata=visualizer_metadata,
        # instance_mode=ColorMode.IMAGE,
        instance_mode=0,  # TODO: replace this hard-coded flag
    )

    for image_file in tqdm(files_to_process):
        
        # load image
        if not isinstance(image_file, Path):
            image_file = Path(image_file)
        assert image_file.exists(), f"File {image_file} does not exist"
        image = Image.open(image_file)

        # TODO: implement TTA if necessary
        # run panoptic segmentation inference and post-process the outputs
        processed_data = process_image(image, processor, model, coco_to_scannet)
        assert len(processed_data) == 1, "Only one image should be processed at a time"

        # visualize results
        visualizer = visualizer_func(np.array(image))

        panoptic_seg, segments_info, _, _ = processed_data[0]["panoptic_seg"]
        vis_output = visualizer.draw_panoptic_seg_predictions(
            panoptic_seg.to("cpu"), segments_info,
        )

        # save outputs
        save_panoptic(
            processed_data[0], processed_data[0], panoptic_dir / f"{image_file.stem}.ptz",
        )
        vis_output.save(vis_dir / f"{image_file.stem}.png")


def load_oneformer(pretrained_model_name: str) -> Tuple[OneFormerProcessor, OneFormerForUniversalSegmentation]:
    """Loads OneFormer model and processor"""
    processor = OneFormerProcessor.from_pretrained(
        pretrained_model_name
    )
    model = OneFormerForUniversalSegmentation.from_pretrained(
        pretrained_model_name
    )
    return processor, model

def process_image(
    image: Image.Image,
    oneformer_processor: OneFormerProcessor,
    oneformer_model: OneFormerForUniversalSegmentation,
    label_translator: COCOtoScanNet,
) -> Dict[str, TensorType]:
    """Processes the given image file and returns outputs."""

    # get resolution of the image (required for resizing)    
    width, height = image.size

    # process input image
    inputs = oneformer_processor(image, ["panoptic"], return_tensors="pt")

    # run forward pass
    with torch.no_grad():
        outputs = oneformer_model(**inputs)

    # post-process OneFormer outputs
    panoptic_outputs = process_oneformer_panoptic_outputs(
        outputs["masks_queries_logits"],
        outputs["class_queries_logits"],
        target_sizes=(height, width),
        label_translator=label_translator,
    )

    return panoptic_outputs

@typechecked
def process_oneformer_panoptic_outputs(
    masks_queries_logits: TensorType[1, "Q", "H", "W"],
    class_queries_logits: TensorType[1, "Q", "C"],
    target_sizes: Tuple[int, int],
    label_translator: COCOtoScanNet,
    object_mask_threshold: float = 0.8,
    overlap_threshold: float = 0.8,
) -> List:
    """
    Processses the prediction results from OneFormer.

    Args:
        masks_queries_logits: Tensor of shape (1, Q, H, W) containing the predicted mask logits.
        class_queries_logits: Tensor of shape (1, Q, C) containing the predicted class logits.
        target_sizes: Tuple containing the target image size (height, width).
        label_translator: 

    Returns:
        A dictionary of PyTorch tensors.

        TODO: Add documentation for the items in the dictionary.
    """

    # resize predicted mask
    height, width = target_sizes
    masks_queries_logits = F.interpolate(
        masks_queries_logits,
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    )

    # reduce batch dimension
    masks_queries_logits = masks_queries_logits[0]
    class_queries_logits = class_queries_logits[0]

    # TODO: M2F panoptic inference code goes here
    scannet_thing_ids = label_translator.scannet_thing_ids
    num_scannet_classes = label_translator.num_scannet_classes
    coco_to_scannet = label_translator.coco_to_scannet
    invalid_classes = label_translator.invalid_classes

    # compute per mask class probabilities and per-pixel probabilities
    cur_mask_cls_probabilities, cur_masks = compute_scannet_class_and_mask_probabilities(
        masks_queries_logits,
        class_queries_logits,
        num_scannet_classes,
        coco_to_scannet,
        invalid_classes,
        object_mask_threshold,
    )

    # compute final predictions and pack them into a dictionary
    predictions = post_tta_merging(
        cur_mask_cls_probabilities,
        cur_masks,
        scannet_thing_ids,
        num_scannet_classes,
        overlap_threshold,
    )

    return predictions

@typechecked
def compute_scannet_class_and_mask_probabilities(
    masks_queries_logits: TensorType["Q", "H", "W"],
    class_queries_logits: TensorType["Q", "C"],
    num_scannet_classes: int,
    coco_to_scannet: List[int],
    invalid_classes: List[int],
    object_mask_threshold: float = 0.8,
) -> Tuple[TensorType, TensorType]:
    """
    Brought from Mask2Former repo:
    https://github.com/nihalsid/Mask2Former/blob/main/mask2former/maskformer_model.py#L395
    """
    masks_queries_logits = masks_queries_logits.sigmoid()
    pre_scores, pre_labels = F.softmax(class_queries_logits, dim=-1).max(-1)
    # todo: for now removing (pre_scores > self.object_mask_threshold)

    invalid_coco_label_index = class_queries_logits.shape[-1] - 1
    keep_labels = pre_labels.ne(invalid_coco_label_index) #& (pre_scores > self.object_mask_threshold)

    pre_cur_masks = masks_queries_logits[keep_labels]
    pre_cur_masks_cls = class_queries_logits[keep_labels]
    pre_cur_masks_cls = pre_cur_masks_cls[:, :-1]

    cur_mask_cls_scannet = pre_cur_masks_cls.clone()
    # -inf for impossible classes
    cur_mask_cls_scannet[:, invalid_classes] = -float('inf')
    smax_cur_mask_cls_scannet = F.softmax(cur_mask_cls_scannet, dim=-1)
    # 0 impossible classes after softmax
    # smax_cur_mask_cls_scannet[:, invalid_classes] = 0

    smax_reduced_cur_mask_cls_scannet = torch.zeros(
        [smax_cur_mask_cls_scannet.shape[0], num_scannet_classes],
        device=class_queries_logits.device,
    )
    torch_scatter.scatter_add(
        smax_cur_mask_cls_scannet,
        index=torch.tensor(coco_to_scannet, device=class_queries_logits.device).long(),
        out=smax_reduced_cur_mask_cls_scannet
    )

    scores, labels = smax_reduced_cur_mask_cls_scannet.max(-1)
    # todo: for now removing scores > self.object_mask_threshold
    # keep = scores > self.object_mask_threshold
    # note: everything is kept in the below keep mask, it is all true's
    keep = torch.ones_like(scores > object_mask_threshold).bool()
    cur_masks = pre_cur_masks[keep]
    cur_mask_cls_probabilities = smax_reduced_cur_mask_cls_scannet[keep]
    
    return cur_mask_cls_probabilities, cur_masks

def post_tta_merging(
    cur_mask_cls_probabilities,
    cur_masks,
    scannet_thing_ids,
    num_scannet_classes,
    overlap_threshold: float = 0.8,
):

    processed_results = [{}]
    cur_scores, cur_classes = cur_mask_cls_probabilities.max(-1)
    cur_prob_masks = cur_scores.view(-1, 1, 1) * cur_masks
    h, w = cur_masks.shape[-2:]
    panoptic_seg = torch.zeros((h, w), dtype=torch.int32, device=cur_masks.device)
    pano_seg_probabilities = torch.zeros((h, w, num_scannet_classes), dtype=torch.float32, device=cur_masks.device)
    pano_seg_probabilities[:, :, 0] = 1.
    pano_seg_mask_preds = torch.ones((h, w), dtype=torch.float32, device=cur_masks.device)
    segments_info = []

    current_segment_id = 0

    if cur_masks.shape[0] == 0:
        # We didn't detect any mask :(
        processed_results[-1]["panoptic_seg"] = panoptic_seg, segments_info
    else:
        # take argmax
        cur_mask_ids = cur_prob_masks.argmax(0)
        stuff_memory_list = {}
        for k in range(cur_classes.shape[0]):
            pred_class = cur_classes[k].item()
            isthing = pred_class in scannet_thing_ids
            mask_area = (cur_mask_ids == k).sum().item()
            original_area = (cur_masks[k] >= 0.5).sum().item()
            mask = (cur_mask_ids == k) & (cur_masks[k] >= 0.5)

            if mask_area > 0 and original_area > 0 and mask.sum().item() > 0:

                ####
                # 20230522 seungwoo. remove class variable
                # if mask_area / original_area < self.overlap_threshold:
                if mask_area / original_area < overlap_threshold:
                ####
                    continue

                # merge stuff regions
                if not isthing:
                    if int(pred_class) in stuff_memory_list.keys():
                        panoptic_seg[mask] = stuff_memory_list[int(pred_class)]
                        pano_seg_probabilities[mask, :] = cur_mask_cls_probabilities[k, :]
                        pano_seg_mask_preds[mask] = cur_prob_masks[k, mask]
                        continue
                    else:
                        stuff_memory_list[int(pred_class)] = current_segment_id + 1

                current_segment_id += 1
                panoptic_seg[mask] = current_segment_id
                pano_seg_probabilities[mask, :] = cur_mask_cls_probabilities[k, :]
                pano_seg_mask_preds[mask] = cur_prob_masks[k, mask]

                segments_info.append(
                    {
                        "id": current_segment_id,
                        "isthing": bool(isthing),
                        "category_id": int(pred_class),
                        "score": cur_scores[k].item(),
                    }
                )
        processed_results[-1]["panoptic_seg"] = panoptic_seg, segments_info, pano_seg_probabilities, pano_seg_mask_preds

    return processed_results

def save_panoptic(predictions, predictions_notta, out_filename):
    mask, segments, probabilities, confidences = predictions["panoptic_seg"]
    mask_notta, segments_notta, _, confidences_notta = predictions_notta["panoptic_seg"]

    with gzip.open(out_filename, "wb") as fid:
        torch.save(
            {
                "mask": mask,
                "segments": segments,
                "mask_notta": mask_notta,
                "segments_notta": segments_notta,
                "confidences_notta": confidences_notta,
                "probabilities": probabilities,
                "confidences": confidences,
                # "feats": predictions["res3_feats"]
            }, fid
        )

if __name__ == "__main__":
    main(tyro.cli(Args))