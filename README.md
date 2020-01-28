# Fixation Mapping with Detectron2

Automated fixation mapping based on Tobii Glasses' data with FAIR's Detectron2. 

(to be filled)

## Requirements

- GPU (Inference is super slow on CPU only, thus is not supported in the script)
- Detectron2

  For Linux / MacOS: follow the [official guide](https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md) for installation and dependencies. 

  For windows (unfortunately), follow the build procedure in [this repo](https://github.com/conansherry/detectron2) to build Detectron2 on your machine. 

- cv2

- numpy

- pandas

## Prepare the data

### Eye-tracking Video

The **full stream (i.e. not segment)** video of the participant. This video can be acquired from the live data as fullstream.mp4, or from exporting the entire video from Tobii ProLab. 

### Fixation data table

`.tsv` data table exported with Tobii ProLab. Must include the following columns: `Recording timestamp, Eye movement type, Eye movement type index, Gaze event duration, Fixation point X, Fixation point Y`.  Other columns are optional. 

The fixation data table can be the the data of a segment. 

Raw gaze data is not supported in this version. 

## Run
**Clone this repo first, and make sure everything is there before running the scripts**  


**fixation_mapping.py**

Before running, make sure that the `labels` folder is in the same working directory of the script.

**arguments:**

- Detectron (dp): path to your Detectron2 folder. This is usually in your virtual environment if you use one, or could be in your C:\Users\. 
- VideoFile (vf): path to the full-stream video file.
- DataFile (ff): path to fixation data table (.xlsx)
- OutFolder (of): folder for output file

For example: `$ python fixation_mapping.py C:\Users\marki\detectron2 F:\videos\recording30_full.mp4  F:\exports\recording30_segment.tsv F:\exports\mapped `   

 **Note:** During the execution of the script, a `.pkl` file will be downloaded as the weight for the segmentation model. It will be saved at `./model/model_final_dbfeb4.pkl`

## Output

A `.tsv` data table with each fixation mapped to an object. An example table would look like: 

| Recording timestamp | Eye movement type index | Gaze  event duration | Fixation point X | Fixation point Y | target     |  phone_x |  phone_y |
| ------------------: | ----------------------: | -------------------: | ---------------: | ---------------: | :--------- | -------: | -------: |
|              715321 |                    1514 |                   60 |              765 |              357 | road       |          |          |
|            715521.5 |                    1515 |                  300 |              559 |              308 | road       |          |          |
|            715762.5 |                    1516 |                  140 |              632 |              355 | car        |          |          |
|            716361.5 |                    1517 |                  580 |             1021 |              700 | cell phone | 0.769508 | 0.338248 |

where `target` columns record the object the fixation is mapped to. A complete list of the available objects can be found at `labels/complete_list.txt`  

When a fixations is mapped to a `cell phone` object, the approximate (proportional) location of the fixation on the cell phone is calculated and stored in `phone_x` and `phone_y` columns

## Syncing mobile eye-tracking data with screen-recordings of the mobile stimuli 

(to be filled)

## Source and more

- Detectron2
- Panoptic segmentation
- Coco thing dataset, coco stuff dataset
- Tobii ProLab manual
