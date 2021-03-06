# ALPR from videos for parking vehicles

## Introduction

This repo is based on the `alpr-unconstrained` work 
["License Plate Detection and Recognition in Unconstrained 
Scenarios"](https://github.com/sergiomsilva/alpr-unconstrained)
published by Silva and Jung (2018). 
Their work addresses the low performance issue 
of other open source ALPR (e.g. openALPR) 
when license plate image is not in front view.

This repo is the core AI submodule of our 
["illegal parking detection system"](https://devpost.com/software/cvparking) project.
It extract license plate numbers from 
patrolling video and estimates all vehicles geographic coordinates. 


## Requirements

Environment
- Linux (tested on Ubuntu 16.04)
- Python 2.7

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

Run the following command to download trained model from 
the work by Silva and Jung (2018)

```shellscript
$ bash get-networks.sh
```

## Run
**All input output file path will be relative to the 
folder containing this repo**

You can run either through IDE (e.g. PyCharm) or command line. 

First, run `run_alpr.py` to detect license plate in each given time step
- `time` unit is second
    - `start_time`: start time to capture video frame
    - `time_step`: the time step to capture next frame
- `landscape`: whether video is taken in landscape (1) or portrait (0)
- `height_ratio`: the minimum ratio of vehicle height against video image
height. Those qualified will be considered clear and large enough 
for license plate detection.
- `focal_length`: focal length of video camera (mm)
- `sensor_height`: video camera sensor height (mm)
- `veh_height`: a constant vehicle height for distance estimation (mm)
- `out_f_lprs`:  name of output file contains detected plate number and 
vehicle distance at each video `time`
- `out_dir`: relative path of folder to store all processed output 
and `out_f_lprs`. 
- `test_f`: input video file relative path

Note: `out_dir`, `test_f` must be relative to the parent 
directory of this repo.

```shellscript
$ start_time=0
$ time_step=1
$ landscape=1
$ height_ratio=0.4
$ focal_length=4.44
$ sensor_height=4.29
$ veh_height=1600.0
$ out_f="all_frames_lps.csv"
$ out_dir="tmp"
$ test_f="test.mp4"
$ python src/run_alpr.py $start_time $time_step $landscape $height_ratio \
$  $focal_length $sensor_height $veh_height $out_f_lprs $out_dir $test_f
```

Second, run `extract_lprs.py` to extract valid video license plates along with 
its time of appearing and disappearing, as well as the vehicle location in 
lat and lon.

- `camera_angle`: the counter-clockwise angle from 
camera moving trajectory to camera view sight line in degree 
- `view_dist_ext`: a small distance offset from back of car to center
- `char_sim`: minimum similarity threshold for two plate numbers 
to be similar to each other. For each plate number, 
the most similar other one is assumed to be referring to a same vehicle.
This parameter is used to reduce false positive of plate number 
recognition.
- `f_lprs`: relative path to output file from `run_alpr.py` containing 
license plate and vehicle distance at each video time
- `f_gps`: relative path to patrol gps, 
currently only accept start and end points, asusming straight line
between samples
- `lpr_out_f`: relative path of output file containing 
the summarized valid plate number and estimate vehicle location
- `traj_out_f`: relative path of output file containg interpolated patrol trajectory

Note: `f_lprs`, `f_gps`, `lpr_out_f`, `traj_out_f` must be relative to 
the parent directory of this repo.

```shellscript
$ camera_angle=70
$ view_dist_ext = 1.0
$ char_sim=0.5
$ f_lprs='tmp/all_frames_lps.csv'
$ f_gps='test.gps'
$ lpr_out_f='test.lpr'
$ traj_out_f='test.traj'
$ python src/extract_lprs.py $camera_angle $view_dist_ext $char_sim $f_lprs \
$  $f_gps $lpr_out_f $traj_out_f
```