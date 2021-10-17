# Object Counting using YOLOv3, Deep Sort and Tensorflow
This repository implements a modified version of theAIGuysCode/yolov3_deepsort, using YOLOv3 and DeepSORT in order to perfrom real-time object counting. 

![Demo of Car Counter](data/helpers/roundabout_Results.gif)

## Getting started

#### Recommended setup and running by Haantzy
```bash
# Tensorflow GPU
conda env create -f conda-gpu.yml
conda activate tracker-gpu

# yolov3
wget https://pjreddie.com/media/files/yolov3.weights -O weights/yolov3.weights

# yolov3
python load_weights.py

#While in the main foler
python object_tracker_counter.py --video ./data/video/roundabout.mp4
```

#### Default Steps from object detector
#### Conda (Recommended)

```bash
# Tensorflow CPU
conda env create -f conda-cpu.yml
conda activate tracker-cpu

# Tensorflow GPU
conda env create -f conda-gpu.yml
conda activate tracker-gpu
```

#### Pip
```bash
# TensorFlow CPU
pip install -r requirements.txt

# TensorFlow GPU
pip install -r requirements-gpu.txt
```

### Nvidia Driver (For GPU, if you haven't set it up already)
```bash
# Ubuntu 18.04
sudo add-apt-repository ppa:graphics-drivers/ppa
sudo apt install nvidia-driver-430
# Windows/Other
https://www.nvidia.com/Download/index.aspx
```
### Downloading official pretrained weights
For Linux: Let's download official yolov3 weights pretrained on COCO dataset. 

```
# yolov3
wget https://pjreddie.com/media/files/yolov3.weights -O weights/yolov3.weights

# yolov3-tiny
wget https://pjreddie.com/media/files/yolov3-tiny.weights -O weights/yolov3-tiny.weights
```

For Windows:
You can download the yolov3 weights by clicking [here](https://pjreddie.com/media/files/yolov3.weights) and yolov3-tiny [here](https://pjreddie.com/media/files/yolov3-tiny.weights) then save them to the weights folder.
  
### Saving your yolov3 weights as a TensorFlow model.
Load the weights using `load_weights.py` script. This will convert the yolov3 weights into TensorFlow .tf model files!

```
# yolov3
python load_weights.py

# yolov3-tiny
python load_weights.py --weights ./weights/yolov3-tiny.weights --output ./weights/yolov3-tiny.tf --tiny
```

After executing one of the above lines, you should see proper .tf files in your weights folder. You are now ready to run object tracker.

## Running the Object Counter
Now you can run the object counter for whichever model you have created, pretrained, tiny, or custom.
```
# yolov3 on video
python object_tracker_counter.py --video ./data/video/roundabout.mp4

#yolov3-tiny 
python object_tracker_counter.py --video ./data/video/roundabout.mp4 --weights ./weights/yolov3-tiny.tf --tiny
```

## Check out these other videos demonstrating cars being counted from a variety of sources


## Acknowledgments from Haantzy
* [Object Tracking using YOLOv3, Deep Sort and Tensorflow](https://github.com/theAIGuysCode/yolov3_deepsort)
## Orignial Acknowledgments
* [Yolov3 TensorFlow Amazing Implementation](https://github.com/zzh8829/yolov3-tf2)
* [Deep SORT Repository](https://github.com/nwojke/deep_sort)
* [Yolo v3 official paper](https://arxiv.org/abs/1804.02767)
