import os
import sys
sys.path.insert(0, './imsearch')
from descripter import ColorDescriptor
from searcher import Searcher
import cv2
import glob
import pandas as pd
import argparse

parser = argparse.ArgumentParser(description='do something together with screen recording video')
parser.add_argument('TargetImagesFolder', metavar='im', type=str, help='')
parser.add_argument('ScreenVideo', metavar='sr', type=str, help='Path to screen recording video')
parser.add_argument('FixationsFile', metavar='fx', type=str, help='Path to mapped fixation file xlsx')
parser.add_argument('TimeOffset', metavar='t', type=int, help='time offset in milliseconds')
parser.add_argument('--index', metavar='-i', action='store', dest=index_path, type=str)
parser.add_argument('--binsize', metavar='-b', action='store', dest=binsizeinput, nargs='+', type=int)
args = parser.parse_args()
pics_path = args.TargetImagesFolder
video_path = args.ScreenVideo
fixation_path = args.FixationFile
time_offset = args.TimeOffset
# todo: the optionals
if args.index:
    # if there is index, don't build it, skip to query directly
    if_build_index = False
    index_path = args.index
else:
    index_path = os.path.join(pics_path, 'index.csv')
if args.binsize:
    bin_size = tuple(args.binsize)

# arguments: time offset, path to "database", ET data table, phone recoding video
# optional: bin size, index path

# sample arguments
pics_path = ''
index_path = r'F:\Play\transformation_phone_mapping\image search engine\index\index_202030.csv'  # -- optional
bin_size = (20, 20, 30)  # optional
video_path = ''
fixation_path = ''
time_offset = 123  # in miliseconds
if_build_index = True


# build index: independent of query input
def build_index(data_path, index_path, bin_size=(20, 20, 30)):
    cd = ColorDescriptor(bin_size)
    output = open(index_path, 'w')
    # loop through the folder with glob
    for imagePath in glob.glob(data_path + "/*.jpg"):
        # extract the image ID (i.e. the unique filename) from the image path and load the image itself
        imageID = imagePath[imagePath.rfind("/") + 1:]
        image = cv2.imread(imagePath)

        # describe the image
        features = cd.describe(image)

        # write the features to file
        features = [str(f) for f in features]
        output.write("%s,%s\n" % (imageID, ",".join(features)))

    # close the index file
    output.close()


# search: input query image, return best match
def query(query, index_path, limit):
    # note "query" is a cv2 ready image
    cd = ColorDescriptor(bin_size)
    features = cd.describe(query)
    searcher = Searcher(index_path)
    results = searcher.search(features, limit)
    # a little bit parsing
    results_ = [list(item) for item in results]
    for item in results_:
        item[1] = item[1].split('\\')[-1]
    return results_
    return results_


# read df, find row of phone, find timestamp, read phone recording, get image, pin result

def find_query_image(timestamp, offset, video_cap):
    sr_time = timestamp - offset
    frame_no = round(sr_time / 1000 * fps)
    video_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
    ret, frame = cap.read()
    return frame


# do
df = pd.read_excel(fixation_path)
sr_video = cv2.VideoCapture(video_path)
sr_fps = sr_video.get(cv2.CAP_PROP_FPS)
if if_build_index:
    build_index(pics_path, index_path, bin_size)  # only build index when there is no index yet

# loop fixation df to fill best match if fixation is on phone
for i in df.index:
    if df.at[i, 'target'] == 'cell phont':
        et_timestamp = df.at[i, 'Recording timestamp']
        phone_image = find_query_image(et_timestamp, time_offset, sr_video)
        best_match = query(phone_image, index_path, 3)[0]
        df.at[i, 'best_match'] = best_match
    else:
        continue

filename = os.path.basename(fixaton_path)
filedir = os.path.dirname(fixation_path)
df.to_excel(os.path.join(filedir, 'screen_' + filename))

# finished
