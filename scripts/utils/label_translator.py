"""
label_translator.py

A collection of classes and functions for translating labels between different datasets.
"""

from collections import OrderedDict
import csv
from pathlib import Path
from typing import Dict, Union

try:
    from scripts.utils.constants import COCO_TO_SCANNET_DIR, SCANNET_THINGS_DIR
except ImportError:
    print(
        "Import error occured. Possibly PYTHONPATH is not set. Please set PYTHONPATH to root of this repository."
    )

class COCOtoScanNet:
    
    def __init__(self, is_reduced_scannet: bool = False):

        # determine the CSV files to load
        data_tag = "reduced" if is_reduced_scannet else "extended"
        coco_to_scannet_csv_file = COCO_TO_SCANNET_DIR / f"coco_to_scannet_{data_tag}.csv"
        things_csv_file = SCANNET_THINGS_DIR / f"scannet_{data_tag}_things.csv"
        assert coco_to_scannet_csv_file.exists(), f"File does not exist: {str(coco_to_scannet_csv_file)}"
        assert things_csv_file.exists(), f"File does not exist: {str(things_csv_file)}"

        self.coco_to_scannet_csv_file = coco_to_scannet_csv_file
        """A path to CSV file holding conversion from COCO to ScanNet labels."""
        self.things_csv_file = things_csv_file
        """A path to CSV file holding COCO thing classes."""

        # load a mapping table from CSV file
        self.scannet_thing_ids, self.num_scannet_classes = self._load_thing_semantics()

        # load thing classes from CSV file
        self.coco_to_scannet, self.invalid_classes  = self._load_coco_to_scannet()

        # load class names from CSV file
        self.scannet_class_names = self._load_scannet_class_names()

    def _load_thing_semantics(self):
        thing_semantics = [False]
        for cllist in [x.strip().split(',') for x in Path(str(self.things_csv_file)).read_text().strip().splitlines()]:
            thing_semantics.append(bool(int(cllist[1])))
        return [i for i in range(len(thing_semantics)) if thing_semantics[i]], len(thing_semantics)

    def _load_coco_to_scannet(self):
        """
        Builds a mapping table from COCO -> ScanNet.
        """
        coco_to_scannet = []
        invalid_classes = []
        # for cidx, cllist in enumerate([x.strip().split(',') for x in Path(f"coco_to_scannet_{sc_version}.csv").read_text().strip().splitlines()]):
        for cidx, cllist in enumerate([x.strip().split(',') for x in Path(str(self.coco_to_scannet_csv_file)).read_text().strip().splitlines()]):
            if int(cllist[1]) != -1:
                coco_to_scannet.append(int(cllist[1]))
            else:
                invalid_classes.append(cidx)
                coco_to_scannet.append(0)
        return coco_to_scannet, invalid_classes

    def _load_scannet_class_names(self):
        class_names = ["none"]
        # for cllist in [x.strip().split(',') for x in Path(f"scannet_{sc_version}_things.csv").read_text().strip().splitlines()]:
        for cllist in [x.strip().split(',') for x in Path(str(self.things_csv_file)).read_text().strip().splitlines()]:
            class_names.append(cllist[0])
        return class_names