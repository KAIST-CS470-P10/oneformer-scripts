"""
oneformer_post_processor.py

A collection of utility functions borrowed from 
https://github.com/huggingface/transformers/blob/v4.29.1/src/transformers/models/oneformer/image_processing_oneformer.py
"""

from typing import Dict, List, Set, Optional, Tuple

import torch
import torch.nn as nn
from torchtyping import TensorType

# Copied from transformers.models.detr.image_processing_detr.remove_low_and_no_objects
def remove_low_and_no_objects(masks, scores, labels, probs, object_mask_threshold, num_labels):
    """
    Binarize the given masks using `object_mask_threshold`, it returns the associated values of `masks`, `scores` and
    `labels`.

    Args:
        masks (`torch.Tensor`):
            A tensor of shape `(num_queries, height, width)`.
        scores (`torch.Tensor`):
            A tensor of shape `(num_queries)`.
        labels (`torch.Tensor`):
            A tensor of shape `(num_queries)`.
        object_mask_threshold (`float`):
            A number between 0 and 1 used to binarize the masks.
    Raises:
        `ValueError`: Raised when the first dimension doesn't match in all input tensors.
    Returns:
        `Tuple[`torch.Tensor`, `torch.Tensor`, `torch.Tensor`]`: The `masks`, `scores` and `labels` without the region
        < `object_mask_threshold`.
    """
    if not (masks.shape[0] == scores.shape[0] == labels.shape[0] == probs.shape[0]):
        raise ValueError("mask, scores and labels must have the same shape!")

    to_keep = labels.ne(num_labels) & (scores > object_mask_threshold)

    return masks[to_keep], scores[to_keep], labels[to_keep], probs[to_keep]

# Copied from transformers.models.detr.image_processing_detr.check_segment_validity
def check_segment_validity(mask_labels, mask_probs, k, mask_threshold=0.5, overlap_mask_area_threshold=0.8):
    # Get the mask associated with the k class
    mask_k = mask_labels == k
    mask_k_area = mask_k.sum()

    # Compute the area of all the stuff in query k
    original_area = (mask_probs[k] >= mask_threshold).sum()
    mask_exists = mask_k_area > 0 and original_area > 0

    # Eliminate disconnected tiny segments
    if mask_exists:
        area_ratio = mask_k_area / original_area
        if not area_ratio.item() > overlap_mask_area_threshold:
            mask_exists = False

    return mask_exists, mask_k

# Copied from transformers.models.detr.image_processing_detr.compute_segments
def compute_segments(
    mask_probs,
    pred_scores,
    pred_labels,
    pred_probs,
    mask_threshold: float = 0.5,
    overlap_mask_area_threshold: float = 0.8,
    label_ids_to_fuse: Optional[Set[int]] = None,
    target_size: Tuple[int, int] = None,
):
    height = mask_probs.shape[1] if target_size is None else target_size[0]
    width = mask_probs.shape[2] if target_size is None else target_size[1]

    num_classes = pred_probs.shape[-1]

    segmentation = torch.zeros((height, width), dtype=torch.int32, device=mask_probs.device)
    segments: List[Dict] = []
    probability_map = torch.zeros((height, width, num_classes), dtype=torch.float32, device=mask_probs.device)

    if target_size is not None:
        mask_probs = nn.functional.interpolate(
            mask_probs.unsqueeze(0), size=target_size, mode="bilinear", align_corners=False
        )[0]

    current_segment_id = 0

    # Weigh each mask by its prediction score
    mask_probs *= pred_scores.view(-1, 1, 1)
    mask_labels = mask_probs.argmax(0)  # [height, width]

    # Keep track of instances of each class
    stuff_memory_list: Dict[str, int] = {}
    for k in range(pred_labels.shape[0]):
        pred_class = pred_labels[k].item()
        pred_prob = pred_probs[k]
        should_fuse = pred_class in label_ids_to_fuse

        # Check if mask exists and large enough to be a segment
        mask_exists, mask_k = check_segment_validity(
            mask_labels, mask_probs, k, mask_threshold, overlap_mask_area_threshold
        )

        if mask_exists:
            if pred_class in stuff_memory_list:
                current_segment_id = stuff_memory_list[pred_class]
            else:
                current_segment_id += 1

            # Add current object segment to final segmentation map
            segmentation[mask_k] = current_segment_id
            segment_score = round(pred_scores[k].item(), 6)
            segments.append(
                {
                    "id": current_segment_id,
                    "label_id": pred_class,
                    "was_fused": should_fuse,
                    "score": segment_score,
                }
            )
            if should_fuse:
                stuff_memory_list[pred_class] = current_segment_id

            # Assign class probabilities to the probability map
            row_indices, col_indices = torch.nonzero(mask_k.long(), as_tuple=True)
            probability_map[row_indices, col_indices, :] = pred_prob

    return segmentation, segments, probability_map

# Copied from transformers.models.maskformer.image_processing_maskformer.MaskFormerImageProcessor.post_process_panoptic_segmentation
def post_process_panoptic_segmentation(
    outputs,
    threshold: float = 0.5,
    mask_threshold: float = 0.5,
    overlap_mask_area_threshold: float = 0.8,
    label_ids_to_fuse: Optional[Set[int]] = None,
    target_sizes: Optional[List[Tuple[int, int]]] = None,
) -> List[Dict]:
    """
    Converts the output of [`MaskFormerForInstanceSegmentationOutput`] into image panoptic segmentation
    predictions. Only supports PyTorch.

    Args:
        outputs ([`MaskFormerForInstanceSegmentationOutput`]):
            The outputs from [`MaskFormerForInstanceSegmentation`].
        threshold (`float`, *optional*, defaults to 0.5):
            The probability score threshold to keep predicted instance masks.
        mask_threshold (`float`, *optional*, defaults to 0.5):
            Threshold to use when turning the predicted masks into binary values.
        overlap_mask_area_threshold (`float`, *optional*, defaults to 0.8):
            The overlap mask area threshold to merge or discard small disconnected parts within each binary
            instance mask.
        label_ids_to_fuse (`Set[int]`, *optional*):
            The labels in this state will have all their instances be fused together. For instance we could say
            there can only be one sky in an image, but several persons, so the label ID for sky would be in that
            set, but not the one for person.
        target_sizes (`List[Tuple]`, *optional*):
            List of length (batch_size), where each list item (`Tuple[int, int]]`) corresponds to the requested
            final size (height, width) of each prediction in batch. If left to None, predictions will not be
            resized.

    Returns:
        `List[Dict]`: A list of dictionaries, one per image, each dictionary containing two keys:
        - **segmentation** -- a tensor of shape `(height, width)` where each pixel represents a `segment_id`, set
            to `None` if no mask if found above `threshold`. If `target_sizes` is specified, segmentation is resized
            to the corresponding `target_sizes` entry.
        - **segments_info** -- A dictionary that contains additional information on each segment.
            - **id** -- an integer representing the `segment_id`.
            - **label_id** -- An integer representing the label / semantic class id corresponding to `segment_id`.
            - **was_fused** -- a boolean, `True` if `label_id` was in `label_ids_to_fuse`, `False` otherwise.
                Multiple instances of the same class / label were fused and assigned a single `segment_id`.
            - **score** -- Prediction score of segment with `segment_id`.
    """

    if label_ids_to_fuse is None:
        # logger.warning("`label_ids_to_fuse` unset. No instance will be fused.")
        print("`label_ids_to_fuse` unset. No instance will be fused.")
        label_ids_to_fuse = set()

    class_queries_logits = outputs.class_queries_logits  # [batch_size, num_queries, num_classes+1]
    masks_queries_logits = outputs.masks_queries_logits  # [batch_size, num_queries, height, width]

    batch_size = class_queries_logits.shape[0]
    num_labels = class_queries_logits.shape[-1] - 1

    mask_probs = masks_queries_logits.sigmoid()  # [batch_size, num_queries, height, width]

    # Predicted label and score of each query (batch_size, num_queries)
    class_queries_probs = nn.functional.softmax(class_queries_logits, dim=-1)
    pred_scores, pred_labels = class_queries_probs.max(-1)

    # Loop over items in batch size
    results: List[Dict[str, TensorType]] = []

    for i in range(batch_size):
        mask_probs_item, pred_scores_item, pred_labels_item, class_queries_probs_item = remove_low_and_no_objects(
            mask_probs[i], pred_scores[i], pred_labels[i], class_queries_probs[i], threshold, num_labels,
        )

        # No mask found
        if mask_probs_item.shape[0] <= 0:
            height, width = target_sizes[i] if target_sizes is not None else mask_probs_item.shape[1:]
            segmentation = torch.zeros((height, width)) - 1
            results.append({"segmentation": segmentation, "segments_info": []})
            continue

        # Get segmentation map and segment information of batch item
        target_size = target_sizes[i] if target_sizes is not None else None
        segmentation, segments, probability_map = compute_segments(
            mask_probs=mask_probs_item,
            pred_scores=pred_scores_item,
            pred_labels=pred_labels_item,
            pred_probs=class_queries_probs_item,
            mask_threshold=mask_threshold,
            overlap_mask_area_threshold=overlap_mask_area_threshold,
            label_ids_to_fuse=label_ids_to_fuse,
            target_size=target_size,
        )

        results.append(
            {
                "segmentation": segmentation,
                "segments_info": segments,
                "probability_map": probability_map,
            }
        )
    return results
