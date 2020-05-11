# Fixation Mapping for Mobile Eye-Tracking: to Real-World Objects and Screen Contents

A little tool to process mobile eye-tracking data. It automatically maps fixations to real-world objects and screen contents (when a mobile device is involved in the experiment).

It maps fixations to real-world objects with panoptic segmentation (using FAIR's Detectron2 framework). If a fixation is mapped to object "cell phone", it can be further linked to screen contents on the phone with the help of screen recording videos (content-based image searching). 

**Note:** the current version only supports fixation data collected with Tobii Pro Glasses 2 and exported by Tobii Pro Lab. 

This is a part of MSc. Thesis *A Solution to Analyze Mobile Eye-tracking Data for GI User Research* by Yuhao (Markie) Jiang, ITC-University of Twente, Enschede, The Netherlands, June 2020. 


## Requirements

- Detectron2

  For Linux / MacOS: follow the [official guide](https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md) for installation and dependencies. 

  For windows (unfortunately), follow the build procedure in [this repo](https://github.com/conansherry/detectron2) to build Detectron2 on your machine. 

- cv2

- numpy

- pandas

The scripts are developed and tested on Python 3.7.

## Prepare the data

#### Eye-tracking Video

The **full stream (i.e. not segment)** video of the participant. This video can be acquired from the live data recorder as fullstream.mp4, or by exporting the entire video from Tobii ProLab. 

#### Fixation data table

`.tsv` data table exported with Tobii ProLab. Must include the following columns: `Recording timestamp, Eye movement type, Eye movement type index, Gaze event duration, Fixation point X, Fixation point Y`.  Other columns are optional. 

The fixation data table can be the the data of a segment. 

Raw gaze data is not supported in this version. 

#### Screen recording video and candidate screenshots

The video of screen recording (if applicable) recorded during the experiment. 

The candidate screenshots are the "example" screen contents that you want to compare the frames of screen recording with. It should be the ones that are typical, or the ones of special interest. They should be stored in a folder. `.jpg` and `.png` formats are accepted. 

## Mapping fixations to real-world objects

### Run fixation_mapping.py

Before running, make sure that the `labels` folder is in the same working directory of the script.

**Arguments:**

- Detectron (dp): path to your Detectron2 folder. This is usually in your virtual environment if you use one, or could be in your `C:\Users` if you have installed Detectron2 globally.
- VideoFile (vf): path to the full-stream video file.
- DataFile (ff): path to fixation data table (.tsv)
- OutFolder (of): folder for output file

For example: `$ python fixation_mapping.py C:\\Users\\admin\\detectron2 F:\\videos\\recording30_full.mp4  F:\\exports\\recording30_segment.tsv F:\\exports\\mapped `   

**Note:** During the (first time) execution of the script, a `.pkl` file will be downloaded as the weight for the segmentation model. It will be saved at `./model/model_final_dbfeb4.pkl` 

### Output

A `.tsv` data table with each fixation mapped to an object. An example table would look like: 

| Recording timestamp | Eye movement type index | Gaze  event duration | Fixation point X | Fixation point Y | target     | phone_x | phone_y |
| ------------------: | ----------------------: | -------------------: | ---------------: | ---------------: | :--------- | ------: | ------: |
|              715321 |                    1514 |                   60 |              765 |              357 | road       |         |         |
|            715521.5 |                    1515 |                  300 |              559 |              308 | road       |         |         |
|            715762.5 |                    1516 |                  140 |              632 |              355 | car        |         |         |
|            716361.5 |                    1517 |                  580 |             1021 |              700 | cell phone |    0.76 |    0.33 |

where `target` columns record the object the fixation is mapped to. A complete list of the available objects can be found at `labels/complete_list.txt`  

When a fixations is mapped to a `cell phone` object, the approximate (proportional) location of the fixation on the cell phone is estimated and stored in `phone_x` and `phone_y` columns. 

## Syncing mobile eye-tracking data with screen-recordings of the mobile stimuli 

### Run screen_recording_match.py

**Arguments**

- TargetImagesFolder (im): path to the folders with candidate images
- ScreenVideo (sr): path to the screen recording video
- FixationsFile (fx): path to **mapped** fixation file (`.tsv`)
- TimeOffset (t): time offset in milliseconds (i.e., the time difference between the start of screen-recording video and the start of eye-tracking recording)
- optional: --index (-i): path to an existing index file. If index is already built, use the old index for faster processing time
- optional: --binsize (-b): customize HSV bin size for histogram comparison. Default bin size: (20, 20, 30),  20 bins for H, 20 bins for S and 30 bins for V. 

For example `$ python screen_recording_match.py "F:/videos/image_pool" "F:/videos/screen.mp4" "F:/exports/mapped/tp1_fixaton_mapped.tsv" 201839 ` 

### Output

A `.tsv` data table, when the `target` of the fixation is `cell phone`, one `best_match` image (candidate) will be assigned to the fixation. The output file is saved to the same folder of the input fixation data table. An example table would look like: 

| Recording timestamp | Eye movement type index | Gaze  event duration | Fixation point X | Fixation point Y | target     | phone_x | phone_y | best_match    |
| ------------------: | ----------------------: | -------------------: | ---------------: | ---------------: | :--------- | ------: | ------: | ------------- |
|              715321 |                    1514 |                   60 |              765 |              357 | road       |         |         |               |
|            715521.5 |                    1515 |                  300 |              559 |              308 | road       |         |         |               |
|            715762.5 |                    1516 |                  140 |              632 |              355 | car        |         |         |               |
|            716361.5 |                    1517 |                  580 |             1021 |              700 | cell phone |    0.76 |    0.33 | info_park.jpg |

## Sources and more

- [Detectron2](https://github.com/facebookresearch/detectron2)
- [Panoptic segmentation](https://arxiv.org/abs/1801.00868) (paper)
- [COCO panoptic dataset (API)](https://github.com/cocodataset/panopticapi) and [COCO panoptic task description 2019](http://cocodataset.org/#panoptic-2019)
- [Tobii ProLab manual](https://www.tobiipro.com/siteassets/tobii-pro/user-manuals/Tobii-Pro-Lab-User-Manual/?v=1.138)