<!-- ABOUT THE PROJECT -->
## About The Project
[![Watch the video](https://img.youtube.com/vi/Q8x87n14IUk/maxresdefault.jpg)](https://youtu.be/Q8x87n14IUk)

The panoply of high risk activities in construction works necessitates close supervision and monitoring to ensure that the works are conducted in a safe manner. While most of the monitoring are currently done manually through standing or remote supervision, the increase in quantum of construction works coupled with the reduction in construction labour force calls for the need of a new approach of supervision. With computer vision techniques, close proximity of workers can be detected and highlighted to promote safety of the work place. This project explores the adoption of video analytic solutions including YOLO-based object detection, DeepSORT facial tracking as well distance approximation through stereo cameras.



<!-- GETTING STARTED -->

### Installation

1. Open "yolov5_object_detector" folder
2. Install requirement.txt file for object detector
3. Open "distancing-monitoring-master" folder 
4. Install requirement.txt file for distance estimator

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- USAGE EXAMPLES -->
## Usage
- [ ] Training Yolov5
  - [ ] Open "yolov5_object_detector" folder
  - [ ] Input "train", "test", "val" data images & labels in 
  "Train_Images_2" folder
  - [ ] Run "lta_train.ipynb" notebook, and train on custom images

- [ ] Object Detector
  - [ ] Open "yolov5_object_detector" folder
  - [ ] Run "distance_safety.py" file
  - [ ] Input both expected average height of worker & threshold distance
  
- [ ] Distance Estimator
    - [ ] Open "Construction Site Distance Estimation_Homography" folder
    - [ ] Run "detect.py" file
    - [ ] Run "MergeFrames.py" file
  


<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

* [Dr. Tian Jing](Senior Lecturer & Consultant, Artificial Intelligence Practice)

<p align="right">(<a href="#readme-top">back to top</a>)</p>


