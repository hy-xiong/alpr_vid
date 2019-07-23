# Street parking ALPR for video

## Introduction

This repo is based on the ALPR work at https://github.com/sergiomsilva/alpr-unconstrained

The work of `alpr-unconstrained` have a good performance on detecting 
license plate from various view angles.
This repo uses this work to extract license plates from video and 
detect license plates of all street-parking vehicles.
This repo assumes video is taken from street sidewalk. 

## Requirements

Environment
- Linux: Ubuntu 16.04
- Python: 2.7

Dependencies:
- Keras 2.2.4
- TensorFlow 1.5.0, 
- OpenCV 3.4.2
- NumPy 1.14
- pandas 0.24
- Darknet
  - Provided inside the package. Build it via the following command
  - CPU is enabled by default. Please edit the makefile to run on GPU

```shellscript
$ cd darknet && make
```

Model:

Run the following command to download trained model

```shellscript
$ bash get-networks.sh
```

## Run
First, run `run_alpr.py` to detect license plate in each given time step
- `time` unit is second
    - `start_time`: start time to capture video frame
    - `time_step`: the time step to capture next frame
- `veh_bbox_ratio`: the minimum ratio of a vehicle's bounding box against 
video-frame-image size to assume this vehicle is a street parking vehicle
- `out_f`: base-name of output file recording vehicle license plate
at each video captured video frame 
- `out_dir`: path to the directory to store all processed output 
and `out_f_lprs`
- `video`: path to video file

```shellscript
$ start_time=0
$ time_step=1
$ veh_bbox_ratio=0.2
$ out_f="all_frames_lps"
$ out_dir="tmp"
$ test_f="samples/test.mp4"
$ python run_alpr.py $start_time $time_step $veh_bbox_ratio $out_f $out_dir $video
```

Second, run `extract_lprs.py` to extract valid video license plates along with 
each time they appear and disappear on video. 6-7 character length is considered
potentially valid.
- `n_occur_check`: if the number of captured frames of a detected license 
appear less than this number, do a simple sanity check to see if this 
plate number is correct. 
  - The sanity check go through all other plate numbers which has number of captured
  frames higher than `n_occur_check`. If the similarity between them is higher than 
  a threshold, exclude this plate number.
- `sim_exclude`: The similarity threshold used above
- `f_lprs`: license plate output file from `run_alpr.py`
- `out_f`: path to output file of final detected license plates

```shellscript
$ n_occur_check = 2
$ sim_exclude = 0.5
$ f_lprs = 'tmp/all_frames_lps'
$ out_f = 'tmp/result'
$ python extract_lprs.py $n_occur_check $sim_exclude $f_lprs $out_f
```