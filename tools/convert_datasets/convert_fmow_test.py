import json
import os
import shutil
from tqdm import tqdm
import pdb

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

dst_root = '/p/scratch/hai_ssl4eo/data/fmow/'
src_root = '/p/scratch/hai_ssl4eo/data/fmow/'
map_path = '/p/project/hai_ssl4eo/wang_yi/data/fmow/ground_truth/test_gt_mapping.json'

for cls in CLASSES:
    os.makedirs(dst_root+'test_new/'+cls,exist_ok=True)

with open(map_path) as f:
    data = json.load(f)
    
for img in tqdm(data):
    imgdir = src_root + img['output']
    cls = img['input'].split('/')[1]
    #pdb.set_trace()
    try:
        shutil.move(imgdir,dst_root+'test_new/'+cls)
    except:
        print('Error',imgdir)
        
print('Finished.')