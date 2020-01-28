import pandas as pd
import os
import argparse

parser = argparse.ArgumentParser(description='replace and group detected target values to custom labels')
parser.add_argument('dataFile', metavar='df', type=str, help='Path to mapped fixation file')
args = parser.parse_args()
data_file = args.dataFile
# purpose: replace and group detected target values to custom labels
# case specific, since I don't see the point of it being generalized as Excel can also do this

# group categories: cell phone, building (class), surroundings (class), others
allowed_list = []
building_class = ['window-blind', 'window', 'ceiling', 'building', 'wall',
                  'house', 'wall-brick', 'wall-stone', 'wall-wood']
surrounding_class = ['tree', 'fence', 'sky', 'pavement', 'grass',
                     'dirt', 'rock', 'road', 'river', 'sand',
                     'person', 'bicycle', 'car', 'motorcycle', 'bus',
                     'train', 'truck', 'traffic light', 'stop sign', 'parking meter']
allowed_list.extend(building_class)
allowed_list.extend(surrounding_class)
allowed_list.append('cell phone')

df = pd.read_csv(data_file, delimiter='\t')
# replace things
df.at[~df["target"].isin(allowed_list), "target"] = "others"
df.replace(building_class, 'building', inplace=True)
df.replace(surrounding_class, 'surroundings', inplace=True)
# output
filename = os.path.basename(data_file)
filedir = os.path.dirname(data_file)
df.to_csv(os.path.join(filedir, 'replaced_' + filename), sep='\t')