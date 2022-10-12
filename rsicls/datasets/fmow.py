# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Sequence, Union
import numpy as np

#from .base_dataset import BaseDataset
from .builder import DATASETS
from .custom import CustomDataset

@DATASETS.register_module()
class FMoW(CustomDataset):

    """FMoW dataset.

    - prepare dataset: follow the instructions in https://github.com/EarthNets/Dataset4EO/blob/main/Dataset4EO/datasets/_builtin/fmow.py
    - usage: 
        op1. load data with pytorch dataset (the current version): inheritate `CustomDataset`
        op2. load data with Dataset4EO datapipe: inheritate `EODataset`

    """


    CLASSES = [
        'airport',
        'airport_hangar',
        'airport_terminal',
        'amusement_park',
        'aquaculture',
        'archaeological_site',
        'barn',
        'border_checkpoint',
        'burial_site',
        'car_dealership',
        'construction_site',
        'crop_field',
        'dam',
        'debris_or_rubble',
        'educational_institution',
        'electric_substation',
        'factory_or_powerplant',
        'fire_station',
        'flooded_road',
        'fountain',
        'gas_station',
        'golf_course',
        'ground_transportation_station',
        'helipad',
        'hospital',
        'impoverished_settlement',
        'interchange',
        'lake_or_pond',
        'lighthouse',
        'military_facility',
        'multi-unit_residential',
        'nuclear_powerplant',
        'office_building',
        'oil_or_gas_facility',
        'park',
        'parking_lot_or_garage',
        'place_of_worship',
        'police_station',
        'port',
        'prison',
        'race_track',
        'railway_bridge',
        'recreational_facility',
        'road_bridge',
        'runway',
        'shipyard',
        'shopping_mall',
        'single-unit_residential',
        'smokestack',
        'solar_farm',
        'space_facility',
        'stadium',
        'storage_tank',
        'surface_mine',
        'swimming_pool',
        'toll_booth',
        'tower',
        'tunnel_opening',
        'waste_disposal',
        'water_treatment_facility',
        'wind_farm',
        'zoo'       
    ]

    IMG_EXTENSIONS = ('_msrgb.jpg','_rgb.jpg')

    def __init__(self,
                 data_prefix: str,
                 pipeline: Sequence = (),
                 classes: Union[str, Sequence[str], None] = None,
                 ann_file: Optional[str] = None,
                 test_mode: bool = False,
                 file_client_args: Optional[dict] = None):
        super().__init__(
            data_prefix=data_prefix,
            pipeline=pipeline,
            classes=classes,
            ann_file=ann_file,
            extensions=self.IMG_EXTENSIONS,
            test_mode=test_mode,
            file_client_args=file_client_args)      

