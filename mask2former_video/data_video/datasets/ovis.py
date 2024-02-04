# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Adrián Szlatincsán from ytvis.py

import logging
import numpy as np
import os
from fvcore.common.file_io import PathManager

from detectron2.structures import BoxMode
from detectron2.data import MetadataCatalog
from .ytvis import load_ytvis_json

"""
This file contains functions to parse OVIS dataset of
COCO-format annotations into dicts in "Detectron2 format".
"""

logger = logging.getLogger(__name__)

OVIS_CATEGORIES = [
    {"color": [106, 0, 228], "isthing": 1, "id": 1, "name": "Person"}, 
    {"color": [174, 57, 255], "isthing": 1, "id": 2, "name": "Bird"}, 
    {"color": [255, 109, 65], "isthing": 1, "id": 3, "name": "Cat"}, 
    {"color": [0, 0, 192], "isthing": 1, "id": 4, "name": "Dog"}, 
    {"color": [0, 0, 142], "isthing": 1, "id": 5, "name": "Horse"}, 
    {"color": [255, 77, 255], "isthing": 1, "id": 6, "name": "Sheep"}, 
    {"color": [120, 166, 157], "isthing": 1, "id": 7, "name": "Cow"}, 
    {"color": [209, 0, 151], "isthing": 1, "id": 8, "name": "Elephant"}, 
    {"color": [0, 226, 252], "isthing": 1, "id": 9, "name": "Bear"}, 
    {"color": [179, 0, 194], "isthing": 1, "id": 10, "name": "Zebra"}, 
    {"color": [174, 255, 243], "isthing": 1, "id": 11, "name": "Giraffe"}, 
    {"color": [110, 76, 0], "isthing": 1, "id": 12, "name": "Poultry"}, 
    {"color": [73, 77, 174], "isthing": 1, "id": 13, "name": "Giant_panda"}, 
    {"color": [250, 170, 30], "isthing": 1, "id": 14, "name": "Lizard"}, 
    {"color": [0, 125, 92], "isthing": 1, "id": 15, "name": "Parrot"}, 
    {"color": [107, 142, 35], "isthing": 1, "id": 16, "name": "Monkey"}, 
    {"color": [0, 82, 0], "isthing": 1, "id": 17, "name": "Rabbit"}, 
    {"color": [72, 0, 118], "isthing": 1, "id": 18, "name": "Tiger"}, 
    {"color": [182, 182, 255], "isthing": 1, "id": 19, "name": "Fish"}, 
    {"color": [255, 179, 240], "isthing": 1, "id": 20, "name": "Turtle"}, 
    {"color": [119, 11, 32], "isthing": 1, "id": 21, "name": "Bicycle"}, 
    {"color": [0, 60, 100], "isthing": 1, "id": 22, "name": "Motorcycle"}, 
    {"color": [0, 0, 230], "isthing": 1, "id": 23, "name": "Airplane"}, 
    {"color": [130, 114, 135], "isthing": 1, "id": 24, "name": "Boat"}, 
    {"color": [165, 42, 42], "isthing": 1, "id": 25, "name": "Vehical"}
]

def _get_ovis_instances_meta():
    thing_ids = [k["id"] for k in OVIS_CATEGORIES if k["isthing"] == 1]
    thing_colors = [k["color"] for k in OVIS_CATEGORIES if k["isthing"] == 1]
    assert len(thing_ids) == 25, len(thing_ids)
    # Mapping from the incontiguous OVIS category id to an id in [0, 24]
    thing_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(thing_ids)}
    thing_classes = [k["name"] for k in OVIS_CATEGORIES if k["isthing"] == 1]
    ret = {
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes,
        "thing_colors": thing_colors,
    }
    return ret

if __name__ == "__main__":
    from detectron2.utils.logger import setup_logger
    from detectron2.utils.visualizer import Visualizer, ColorMode, _create_text_labels
    import detectron2.data.datasets  # noqa # add pre-defined metadata
    import sys
    import torch
    import argparse
    from PIL import Image

    logger = setup_logger(name=__name__)
    
    parser = argparse.ArgumentParser(description="maskformer2 demo for builtin configs")
    parser.add_argument(
        "--is-scribble",
        type=bool,
        default=False,
        help="Is it a scribble dataset (True) or not (False)",
    )
    
    args = parser.parse_args()
    
    def extract_frame_dic(dic, frame_idx):
            import copy
            frame_dic = copy.deepcopy(dic)
            annos = frame_dic.get("annotations", None)
            if annos:
                frame_dic["annotations"] = annos[frame_idx]

            return frame_dic
        
    if not args.is_scribble:
        """
        Test the OVIS json dataset loader.
        """
        
        #assert sys.argv[3] in DatasetCatalog.list()
        meta = MetadataCatalog.get("ovis_train")

        json_file = "./datasets/ovis/annotations_train.json"
        image_root = "./datasets/ovis/train"
        dicts = load_ytvis_json(json_file, image_root, dataset_name="ovis_train")
        logger.info("Done loading {} samples.".format(len(dicts)))

        dirname = "ovis-data-vis"
        os.makedirs(dirname, exist_ok=True)

        for d in dicts:
            vid_name = d["file_names"][0].split('/')[-2]
            os.makedirs(os.path.join(dirname, vid_name), exist_ok=True)
            for idx, file_name in enumerate(d["file_names"]):
                img = np.array(Image.open(file_name))
                visualizer = Visualizer(img, metadata=meta)
                vis = visualizer.draw_dataset_dict(extract_frame_dic(d, idx))
                fpath = os.path.join(dirname, vid_name, file_name.split('/')[-1])
                vis.save(fpath)
    else:
        """
        Test the OVIS scribble json dataset loader.
        """
        
        #assert sys.argv[3] in DatasetCatalog.list()
        meta = MetadataCatalog.get("ovis_train")

        json_file = "./datasets/ovis/train_scribble.json"
        image_root = "./datasets/ovis/train"
        dicts = load_ytvis_json(json_file, image_root, dataset_name="ovis_train")
        logger.info("Done loading {} samples.".format(len(dicts)))

        dirname = "ovis-scribble-data-vis"
        os.makedirs(dirname, exist_ok=True)
        
        def _create_text_labels(classes, scores, class_names, is_crowd=None):
            """
            Args:
                classes (list[int] or None)
                scores (list[float] or None)
                class_names (list[str] or None)
                is_crowd (list[bool] or None)

            Returns:
                list[str] or None
            """

            labels = None
            if classes is not None:
                if class_names is not None and len(class_names) > 0:
                    #labels = [class_names[i] for i in classes ]
                    labels = []
                    for i in classes:
                        if i is None:
                            labels.append('background')
                        else:
                            labels.append(class_names[i] if i < len(class_names) else str(i))
                        
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
        
        class ScribbleVisualizer(Visualizer):
            def draw_dataset_dict(self, dic):
                """
                Draw annotations/segmentations in Detectron2 Dataset format.

                Args:
                    dic (dict): annotation/segmentation data of one image, in Detectron2 Dataset format.

                Returns:
                    output (VisImage): image object with visualizations.
                """
                annos = dic.get("annotations", None)
                if annos:
                    if "segmentation" in annos[0]:
                        masks = [x["segmentation"] for x in annos]
                    else:
                        masks = None
                    if "keypoints" in annos[0]:
                        keypts = [x["keypoints"] for x in annos]
                        keypts = np.array(keypts).reshape(len(annos), -1, 3)
                    else:
                        keypts = None

                    boxes = [
                        BoxMode.convert(x["bbox"], x["bbox_mode"], BoxMode.XYXY_ABS)
                        if len(x["bbox"]) == 4
                        else x["bbox"]
                        for x in annos
                    ]

                    colors = None
                    category_ids = [x["category_id"] for x in annos]
                    if self._instance_mode == ColorMode.SEGMENTATION and self.metadata.get("thing_colors"):
                        colors = [
                            self._jitter([x / 255 for x in self.metadata.thing_colors[c]])
                            for c in category_ids
                        ]
                    names = self.metadata.get("thing_classes", None)
                    labels = _create_text_labels(
                        category_ids,
                        scores=None,
                        class_names=names,
                        is_crowd=[x.get("iscrowd", 0) for x in annos],
                    )
                    self.overlay_instances(
                        labels=labels, boxes=boxes, masks=masks, keypoints=keypts, assigned_colors=colors
                    )

                sem_seg = dic.get("sem_seg", None)
                if sem_seg is None and "sem_seg_file_name" in dic:
                    with PathManager.open(dic["sem_seg_file_name"], "rb") as f:
                        sem_seg = Image.open(f)
                        sem_seg = np.asarray(sem_seg, dtype="uint8")
                if sem_seg is not None:
                    self.draw_sem_seg(sem_seg, area_threshold=0, alpha=0.5)

                pan_seg = dic.get("pan_seg", None)
                if pan_seg is None and "pan_seg_file_name" in dic:
                    with PathManager.open(dic["pan_seg_file_name"], "rb") as f:
                        pan_seg = Image.open(f)
                        pan_seg = np.asarray(pan_seg)
                        from panopticapi.utils import rgb2id

                        pan_seg = rgb2id(pan_seg)
                if pan_seg is not None:
                    segments_info = dic["segments_info"]
                    pan_seg = torch.tensor(pan_seg)
                    self.draw_panoptic_seg(pan_seg, segments_info, area_threshold=0, alpha=0.5)
                return self.output

        for d in dicts:
            vid_name = d["file_names"][0].split('/')[-2]
            os.makedirs(os.path.join(dirname, vid_name), exist_ok=True)
            for idx, file_name in enumerate(d["file_names"]):
                img = np.array(Image.open(file_name))
                visualizer = ScribbleVisualizer(img, metadata=meta)
                vis = visualizer.draw_dataset_dict(extract_frame_dic(d, idx))
                fpath = os.path.join(dirname, vid_name, file_name.split('/')[-1])
                vis.save(fpath)