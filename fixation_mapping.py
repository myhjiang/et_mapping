import detectron2
from detectron2.utils.logger import setup_logger
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg

import pandas as pd
import os
import time
import numpy as np
from numpy.linalg import norm
import cv2
import random
import json
import argparse


parser = argparse.ArgumentParser(description='Fixation mapping for Tobii output with FAIR Detectron2')
parser.add_argument('Detectron', metavar='dp', type=str, help='Path to your detectron package folder')
parser.add_argument('VideoFile', metavar='vf', type=str, help='Path to recording video')
parser.add_argument('DataFile', metavar='ff', type=str, help='Path to tobii fixation data export')
parser.add_argument('OutFolder', metavar='of', type=str, help='Path to output folder')
args = parser.parse_args()
detectron_path = args.Detectron
video_path = args.VideoFile
fixation_file_path = args.DataFile
output_folder = args.OutFolder

"""
# example inputs
detectron_path = r'C:\Users\marki\detectron2'
video_path = r'F:\Play\synch_video_data\recording30_full.mp4'
fixaton_file_path = r'F:\Play\synch_video_data\Xiaoling Wang_GeoFARA Metrics.xlsx'
output_foler = ''
"""

# get path to model
cwd = os.getcwd()
model_path = os.path.join(cwd, 'model')
# setup detectron logger
setup_logger()
# get config and model
cfg = get_cfg()
cfg.merge_from_file(detectron_path + '\configs\COCO-PanopticSegmentation\panoptic_fpn_R_50_1x.yaml')
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
cfg.MODEL.WEIGHTS = model_path + '\model_final_dbfeb4.pkl'
predictor = DefaultPredictor(cfg)

# read label json
with open('labels/coco_stuff_rev.json') as json_file:
    stuff_rev = json.load(json_file)
    stuff_rev = {int(k): v for k,v in stuff_rev.items()}
with open('labels/coco_thing_rev.json') as json_file:
    thing_rev = json.load(json_file)
    thing_rev = {int(k): v for k, v in thing_rev.items()}


# process video and map fixation
def map_fixation(video_path, fixation_path):
    """function to process video and map fixation"""
    df = pd.read_excel(fixation_path)
    print('start to map fixations, total fixations to map:', len(df))

    fixation_df = df[df['Eye movement type'] == 'Fixation']
    # mid frame timestamp of fixation, like in manual mapping
    mid_fixation = fixation_df.groupby(['Eye movement type index']).median().reset_index()
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    print('fps:', fps)

    # loop through fixation and do mapping
    time_offset = df.iloc[0]['Recording timestamp']  # segment frame offset
    # todo: where to start time?
    mid_fixation['target'] = ""
    mid_fixation['phone_x'] = ""
    mid_fixation['phone_y'] = ""
    for i in mid_fixation.index:
        start_time = time.time()
        timestamp = mid_fixation.at[i, 'Recording timestamp']
        # fix_x = mid_fixation.get_value(i,'fix_x')
        fix_x = mid_fixation.at[i, 'Fixation point X']
        fix_y = mid_fixation.at[i, 'Fixation point Y']
        # frame_no = round(((timestamp - time_offset) / 1000) * fps)  # if the video is also segment
        frame_no = round((timestamp / 1000) * fps)  # for full video
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
        ret, frame = cap.read()  # between 300ms and 1300ms...?!
        end_1 = time.time()
        # do segmentation
        outputs = predictor(frame)
        segments = outputs['panoptic_seg'][0]
        categories = outputs['panoptic_seg'][1]
        end_2 = time.time()
        segmentation_time = end_2 - end_1  # about 200ms, up to 600
        # locate fixation
        masks = segments.cpu().numpy()  # shape (1080, 1920)
        masks_.append(masks)
        segment_id = masks[int(fix_y), int(fix_x)]  # !!!
        category_id = categories[segment_id - 1]['category_id']
        is_thing = categories[segment_id - 1]['isthing']
        # category id to name
        if is_thing:
            try:
                category = thing_rev[category_id]
                # if it's phone, map to phone coordinates
                if category == 'cell phone':
                    mask = (masks == segment_id)
                    fixation = np.array([int(fix_x), int(fix_y)])  # order!
                    phone_x, phone_y = fixation_to_phone_coords(mask, fixation)
                    mid_fixation.at[i, 'phone_x'] = phone_x
                    mid_fixation.at[i, 'phone_y'] = phone_y
            except:
                print(frame_no, category_id, 'not labelled')
                category = 'unlabeled'
                continue
        if not is_thing:
            try:
                category = stuff_rev[category_id]
            except:
                print(frame_no, category_id, 'not labelled')
                category = 'unlabeled'
                continue
        end_3 = time.time()
        print('fixation mapped', i, ', time cost: ', end_3-start_time)
        # add to df
        mid_fixation.at[i, 'target'] = category

    # only use subset
    mid_fixation = mid_fixation[['Recording timestamp', 'Eye movement type index',
                                 'Fixation point X', 'Fixation point Y', 'target', 'phone_x', 'phone_y']]

    return mid_fixation


# map to phone coordinates
def fixation_to_phone_coords(mask, fixation):
    """function to map fixation to approximate phone coordinates"""
    """not using affine transformation because:
    a) the size of each phone is different 
    b) affine based on the corners introduces more error in mapping that fixation
    """
    mask_im = mask.astype(np.uint8) * 255

    try:
        contours, _, = cv2.findContours(mask_im, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    except:
        _, contours, _, = cv2.findContours(mask_im, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    rect = cv2.minAreaRect(contours[0])
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    bottom_point = np.array(box[0])
    left_point = np.array(box[1])
    right_point = np.array(box[3])
    top_point = np.array(box[2])
    left_edge = norm(left_point - bottom_point)
    right_edge = norm(right_point - bottom_point)
    if left_edge > right_edge:  # assume all phones are rectangular
        # readability problems here...
        p3 = bottom_point
        p2 = left_point
        p1 = top_point
        distance_12 = norm(np.cross(p2 - p1, p1 - fixation)) / norm(p2 - p1)  # on the shorter edge
        distance_23 = norm(np.cross(p2 - p3, p3 - fixation)) / norm(p2 - p3)  # on the longer edge
        phone_x = distance_12 / right_edge
        phone_y = distance_23 / left_edge
    elif left_edge < right_edge:
        p3 = top_point
        p2 = left_point
        p1 = bottom_point
        distance_12 = norm(np.cross(p2 - p1, p1 - fixation)) / norm(p2 - p1)
        distance_23 = norm(np.cross(p2 - p3, p3 - fixation)) / norm(p2 - p3)
        phone_x = distance_12 / left_edge
        phone_y = distance_23 / right_edge
    else:
        print('detect funny phone shape')
        phone_x = -1
        phone_y = -1

    return phone_x, phone_y


# run mapping
masks_ = []  # for function use
mapped_fixation = map_fixation(video_path, fixaton_file_path)
output_name = 'mapped_' + os.path.basename(fixaton_file_path)
mapped_fixation.to_excel(os.path.join(output_folder, output_name))


print('Fixation mapping finished.')

# todo: change the print time thing in mapping function
# todo: one at a time or batch?
