import os
import sys
sys.path.insert(0, './imsearch')
from descripter import ColorDescriptor
from searcher import Searcher
import cv2
import glob
import pandas as pd
import argparse

parser = argparse.ArgumentParser(description='sync with screen recording video and map fixations to phone content')
parser.add_argument('TargetImagesFolder', metavar='im', type=str, help='The folders with candidate images')
parser.add_argument('ScreenVideo', metavar='sr', type=str, help='Path to screen recording video')
parser.add_argument('FixationsFile', metavar='fx', type=str, help='Path to mapped fixation file xlsx')
parser.add_argument('TimeOffset', metavar='t', type=int, help='time offset in milliseconds')
parser.add_argument('--index', metavar='-i', nargs='?', action='store', dest='index_path', type=str, default='none',
                    description='if index is already built, use the old index to save time')
parser.add_argument('--binsize', metavar='-b', action='store', dest='bin_size', nargs='+', type=int,
                    description='customize HSV bin size for histogram comparision')

args = parser.parse_args()
pics_path = args.TargetImagesFolder
video_path = args.ScreenVideo
fixation_path = args.FixationsFile
time_offset = args.TimeOffset
index_path = args.index_path
bin_size = args.bin_size

# optional bin size and tuples
if bin_size is None:
    bin_size = (20, 20, 30)
else:
    bin_size = tuple(bin_size)

# arguments: time offset, path to "database", ET data table, phone recoding video
# optional: bin size, index path

# # sample arguments
# pics_path = ''
# index_path = r'F:\Play\transformation_phone_mapping\image search engine\index\index_202030.csv'  # -- optional
# bin_size = (20, 20, 30)  # optional
# video_path = ''
# fixation_path = ''
# time_offset = 123  # in miliseconds


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
    print(sr_time)
    frame_no = round(sr_time / 1000 * sr_fps)
    print(frame_no)
    video_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
    ret, frame = video_cap.read()
    return frame


# do
df = pd.read_csv(fixation_path, delimiter='\t')
df['best_match'] = ''
df['screentime'] = 0
sr_video = cv2.VideoCapture(video_path)
sr_fps = sr_video.get(cv2.CAP_PROP_FPS)

# only build index when there is no index yet
if index_path == 'none':
    index_path = os.path.join(pics_path, 'index.csv')
    print(f'building index... index will be stored at {index_path}')
    build_index(pics_path, index_path, bin_size)
else:
    print(f'using index from {index_path}')

# loop fixation df to fill best match if fixation is on phone
print('mapping fixation to screen recording contents...')
for i in df.index:
    if df.at[i, 'target'] == 'cell phone':
        et_timestamp = df.at[i, 'Recording timestamp']
        df.at[i, 'screentime'] = et_timestamp - time_offset
        phone_image = find_query_image(et_timestamp, time_offset, sr_video)
        if phone_image is None:
            break
        best_match = query(phone_image, index_path, 3)[0][1]
        df.at[i, 'best_match'] = best_match
    else:
        continue

filename = os.path.basename(fixation_path)
filedir = os.path.dirname(fixation_path)
df.to_csv(os.path.join(filedir, 'screen_' + filename), sep='\t')

print(f"mapping finished, file stored at {os.path.join(filedir, 'screen_' + filename)}")
# finished
