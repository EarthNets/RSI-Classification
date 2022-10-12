# Copyright (c) OpenMMLab. All rights reserved.

import numpy as np

#from .base_dataset import BaseDataset
from .builder import DATASETS
from .multi_label import MultiLabelDataset

@DATASETS.register_module()
class BigEarthNet(MultiLabelDataset):
    """BigEarthNet dataset.
    The `BigEarthNet <https://bigearth.net/>`__
    dataset is a dataset for multilabel remote sensing image scene classification.

    - prepare dataset: follow the instructions in https://github.com/EarthNets/Dataset4EO/blob/main/Dataset4EO/datasets/_builtin/bigearthnet.py
    - usage: 
        op1. load data with pytorch dataset (the current version): inheritate `MultiLabelDataset`
        op2. load data with Dataset4EO datapipe: inheritate `EODataset`

    """

    class_sets = {
        19: [
            "Urban fabric",
            "Industrial or commercial units",
            "Arable land",
            "Permanent crops",
            "Pastures",
            "Complex cultivation patterns",
            "Land principally occupied by agriculture, with significant areas of"
            " natural vegetation",
            "Agro-forestry areas",
            "Broad-leaved forest",
            "Coniferous forest",
            "Mixed forest",
            "Natural grassland and sparsely vegetated areas",
            "Moors, heathland and sclerophyllous vegetation",
            "Transitional woodland, shrub",
            "Beaches, dunes, sands",
            "Inland wetlands",
            "Coastal wetlands",
            "Inland waters",
            "Marine waters",
        ],
        43: [
            "Agro-forestry areas",
            "Airports",
            "Annual crops associated with permanent crops",
            "Bare rock",
            "Beaches, dunes, sands",
            "Broad-leaved forest",
            "Burnt areas",
            "Coastal lagoons",
            "Complex cultivation patterns",
            "Coniferous forest",
            "Construction sites",
            "Continuous urban fabric",
            "Discontinuous urban fabric",
            "Dump sites",
            "Estuaries",
            "Fruit trees and berry plantations",
            "Green urban areas",
            "Industrial or commercial units",
            "Inland marshes",
            "Intertidal flats",
            "Land principally occupied by agriculture, with significant areas of"
            " natural vegetation",
            "Mineral extraction sites",
            "Mixed forest",
            "Moors and heathland",
            "Natural grassland",
            "Non-irrigated arable land",
            "Olive groves",
            "Pastures",
            "Peatbogs",
            "Permanently irrigated land",
            "Port areas",
            "Rice fields",
            "Road and rail networks and associated land",
            "Salines",
            "Salt marshes",
            "Sclerophyllous vegetation",
            "Sea and ocean",
            "Sparsely vegetated areas",
            "Sport and leisure facilities",
            "Transitional woodland/shrub",
            "Vineyards",
            "Water bodies",
            "Water courses",
        ],
    }

    CLASSES = class_sets[19]


    def load_annotations(self):
        assert isinstance(self.ann_file, str)
        data_infos = []
        with open(self.ann_file) as f:
            samples = [x.strip().split(',') for x in f.readlines()]
            for sample in samples:
                filename = sample[0] + '.png'
                gt_label = sample[1:]
                target = np.zeros(19, dtype=np.uint8)
                for id in gt_label:
                    target[int(id)] = 1
                info = {'img_prefix': self.data_prefix}
                info['img_info'] = {'filename': filename}
                info['gt_label'] = np.array(target, dtype=np.uint8)
                data_infos.append(info)
            return data_infos        

