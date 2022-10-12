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
    Dataset features:
    * 590,326 patches from 125 Sentinel-1 and Sentinel-2 tiles
    * Imagery from tiles in Europe between Jun 2017 - May 2018
    * 12 spectral bands with 10-60 m per pixel resolution (base 120x120 px)
    * 2 synthetic aperture radar bands (120x120 px)
    * 43 or 19 scene classes from the 2018 CORINE Land Cover database (CLC 2018)
    Dataset format:
    * images are composed of multiple single channel geotiffs
    * labels are multiclass, stored in a single json file per image
    * mapping of Sentinel-1 to Sentinel-2 patches are within Sentinel-1 json files
    * Sentinel-1 bands: (VV, VH)
    * Sentinel-2 bands: (B01, B02, B03, B04, B05, B06, B07, B08, B8A, B09, B11, B12)
    * All bands: (VV, VH, B01, B02, B03, B04, B05, B06, B07, B08, B8A, B09, B11, B12)
    * Sentinel-2 bands are of different spatial resolutions and upsampled to 10m
    Dataset classes (43):
    0. Agro-forestry areas
    1. Airports
    2. Annual crops associated with permanent crops
    3. Bare rock
    4. Beaches, dunes, sands
    5. Broad-leaved forest
    6. Burnt areas
    7. Coastal lagoons
    8. Complex cultivation patterns
    9. Coniferous forest
    10. Construction sites
    11. Continuous urban fabric
    12. Discontinuous urban fabric
    13. Dump sites
    14. Estuaries
    15. Fruit trees and berry plantations
    16. Green urban areas
    17. Industrial or commercial units
    18. Inland marshes
    19. Intertidal flats
    20. Land principally occupied by agriculture, with significant
        areas of natural vegetation
    21. Mineral extraction sites
    22. Mixed forest
    23. Moors and heathland
    24. Natural grassland
    25. Non-irrigated arable land
    26. Olive groves
    27. Pastures
    28. Peatbogs
    29. Permanently irrigated land
    30. Port areas
    31. Rice fields
    32. Road and rail networks and associated land
    33. Salines
    34. Salt marshes
    35. Sclerophyllous vegetation
    36. Sea and ocean
    37. Sparsely vegetated areas
    38. Sport and leisure facilities
    39. Transitional woodland/shrub
    40. Vineyards
    41. Water bodies
    42. Water courses
    Dataset classes (19):
    0. Urban fabric
    1. Industrial or commercial units
    2. Arable land
    3. Permanent crops
    4. Pastures
    5. Complex cultivation patterns
    6. Land principally occupied by agriculture, with significant
       areas of natural vegetation
    7. Agro-forestry areas
    8. Broad-leaved forest
    9. Coniferous forest
    10. Mixed forest
    11. Natural grassland and sparsely vegetated areas
    12. Moors, heathland and sclerophyllous vegetation
    13. Transitional woodland, shrub
    14. Beaches, dunes, sands
    15. Inland wetlands
    16. Coastal wetlands
    17. Inland waters
    18. Marine waters
    If you use this dataset in your research, please cite the following paper:
    * https://doi.org/10.1109/IGARSS.2019.8900532
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

