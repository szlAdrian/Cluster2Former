import numpy as np
from torch import is_tensor
from detectron2.utils.colormap import random_color
from detectron2.utils.visualizer import ColorMode, Visualizer, GenericMask#, _create_text_labels

# codes are from detectron2.utils.visualizer

def _create_text_labels(classes, scores, class_names, is_crowd=None):
    """
    Args:
        classes (list[int] or None):
        scores (list[float] or None):
        class_names (list[str] or None):
        is_crowd (list[bool] or None):

    Returns:
        list[str] or None
    """

    labels = None
    if classes is not None:
        if class_names is not None and len(class_names) > 0:
            #labels = [class_names[i] for i in classes ]
            labels = [class_names[i] if i < len(class_names) else str(i) for i in classes ]
        else:
            labels = [str(i) for i in classes]
    if scores is not None:
        if labels is None:
            labels = ["{:.0f}%".format(s * 100) for s in scores]
        else:
            labels = ["{} {:.0f}%".format(l, s * 100) for l, s in zip(labels, scores)]
    if labels is not None and is_crowd is not None:
        labels = [l + ("|crowd" if crowd else "") for l, crowd in zip(labels, is_crowd)]
    return labels
        
class VisualizerGT(Visualizer):
    def __init__(self, img_rgb, metadata=None, scale=1.0, instance_mode=ColorMode.IMAGE):
        super().__init__(
            img_rgb, metadata=metadata, scale=scale, instance_mode=instance_mode
        )
    
    def draw_gt_instances(self, gt, panoptic_on = False):
        """
        Draw instance-level prediction results on an image.

        Args:
            gt (Instances): the input of an instance detection/segmentation
                model. Following fields will be used to draw:
                "gt_boxes", "gt_classes", "gt_masks" (or "gt_masks_rle" or "gt_scribles" or "gt_scrible_masks").
            panoptic_on (bool): Whether it is for panoptic segmentation or not.

        Returns:
            output (VisImage): image object with visualizations.
        """
        boxes = gt.gt_boxes if gt.has("gt_boxes") else None
        scores = gt.scores if gt.has("scores") else None
        classes = gt.gt_classes.tolist() if gt.has("gt_classes") else None
        if panoptic_on:
            labels = _create_text_labels(classes, scores, self.metadata.get("stuff_classes", None))
        else:
            labels = _create_text_labels(classes, scores, self.metadata.get("thing_classes", None))

        if classes is not None:
            bg_idx = None
            for i, c in enumerate(classes):
                if c == -1:
                    bg_idx = i
                    break
            if bg_idx is not None:
                labels[bg_idx] = 'background'

        keypoints = gt.pred_keypoints if gt.has("gt_keypoints") else None

        if gt.has("gt_masks") and (not gt.has("gt_scribles")):
            if is_tensor(gt.gt_masks):
                masks = np.asarray(gt.gt_masks)
            else:
                masks = np.asarray(gt.gt_masks.tensor)
            masks = [GenericMask(x, self.output.height, self.output.width) for x in masks]
        elif gt.has("gt_scribles"):
            masks = np.asarray(gt.gt_scribles)
            masks = [GenericMask(x, self.output.height, self.output.width) for x in masks]
        elif gt.has("gt_scrible_masks"):
            if is_tensor(gt.gt_scrible_masks):
                masks = np.asarray(gt.gt_scrible_masks)
            else:
                masks = np.asarray(gt.gt_scrible_masks.tensor)
            masks = [GenericMask(x, self.output.height, self.output.width) for x in masks]
        else:
            masks = None

        if self._instance_mode == ColorMode.SEGMENTATION and self.metadata.get("thing_colors"):
            colors = [
                self._jitter([x / 255 for x in self.metadata.thing_colors[c]]) for c in classes
            ]
            alpha = 0.8
        else:
            colors = None
            alpha = 0.5

        if self._instance_mode == ColorMode.IMAGE_BW:
            if gt.has("gt_masks") and (not gt.has("gt_scribles")):
                self.output.reset_image(
                    self._create_grayscale_image(
                        (gt.gt_masks.any(dim=0) > 0).numpy()
                        if gt.has("gt_masks")
                        else None
                    )
                )
            elif gt.has("gt_scribles"):
                self.output.reset_image(
                    self._create_grayscale_image(
                        (gt.gt_scribles.any(dim=0) > 0).numpy()
                        if gt.has("gt_scribles")
                        else None
                    )
                )
            elif gt.has("gt_scrible_masks"):
                self.output.reset_image(
                    self._create_grayscale_image(
                        (gt.gt_scrible_masks.any(dim=0) > 0).numpy()
                        if gt.has("gt_scrible_masks")
                        else None
                    )
                )
            alpha = 0.3

        self.overlay_instances(
            masks=masks,
            boxes=boxes,
            labels=labels,
            keypoints=keypoints,
            assigned_colors=colors,
            alpha=alpha,
        )
        return self.output