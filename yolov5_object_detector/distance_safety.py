
import math
from tracemalloc import start
import cv2
import numpy as np
import argparse
import os
import sys
from pathlib import Path
from scipy.spatial import distance as dist


def Euc_dist(x1,x2,y1,y2):
    """
    :param:
    p1,p2 = two points for calculating Euclidean Distance

    :return:
    dst = Eculidean Distance between two 2d points
    """
    dst = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

    return dst

def input_ratio(height,image):
    height = float(height)
    image_w = image.shape[0]
    image_h = image.shape[1]
    ratio = height/image_h

    return ratio

#Apply Transformation Matrix
"""
With floor plan we map out the actual points and apply perspective transform to account for the 
real-world distances
""" 
meter_per_pixel = 20.7/111
pts1 = np.float32([[1,80],[1600,80],[1,900],[1600,900]])
pts_world = np.float32([[0,0],[198,0],[0,454],[198,454]]) * meter_per_pixel
matrix_cam2world = cv2.getPerspectiveTransform(pts1,pts_world)



FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

import torch
import torch.backends.cudnn as cudnn

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync



weights= ROOT / 'best.pt'  # model.pt path(s)
source=ROOT/'Train_Images_2/detect_img'
data=ROOT / 'data/coco128.yaml'
imgsz=(640, 640)  # inference size (height, width)
conf_thres=0.25  # confidence threshold
iou_thres=0.45  # NMS IOU threshold
max_det=1000 # maximum detections per image
device='cpu'  # cuda device, i.e. 0 or 0,1,2,3 or cpu
view_img=True  # show results
save_txt= True # save results to *.txt
save_conf=False  # save confidences in --save-txt labels
save_crop=False  # save cropped prediction boxes
nosave=False  # do not save images/videos
classes=None  # filter by class: --class 0, or --class 0 2 3
visualize=False  # visualize features
update=False  # update all models
project=ROOT / 'runs/detect' # save results to project/name
name='exp'  # save results to project/name
exist_ok=False  # existing project/name ok, do not increment
line_thickness=3  # bounding box thickness (pixels)
augment=False # augmented inference
agnostic_nms=False  # class-agnostic NMS
half=False  # use FP16 half-precision inference
dnn=False # use OpenCV DNN for ONNX inference
hide_labels=False  # hide labels
hide_conf=False  # hide confidences

source = str(source)
save_img = not nosave and not source.endswith('.txt')  # save inference images

#Input Height
height = input("Please input real height of a worker: ")
#Input Threshold Distance
threshold_dist = float (input("Please input threshold Distance: "))
# Load model
device = select_device(device)
model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
stride, names, pt = model.stride, model.names, model.pt
imgsz = check_img_size(imgsz, s=stride)  # check image size

# Dataloader

dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
bs = 1  # batch_size
vid_path, vid_writer = [None] * bs, [None] * bs

save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
(save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir



 # Run inference
model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup
seen, windows, dt = 0, [], [0.0, 0.0, 0.0]
for path, im, im0s, vid_cap, s in dataset:
    t1 = time_sync()
    im = torch.from_numpy(im).to(device)
    im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
    im /= 255  # 0 - 255 to 0.0 - 1.0
    if len(im.shape) == 3:
        im = im[None]  # expand for batch dim
    t2 = time_sync()
    dt[0] += t2 - t1

    # Inference
    visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
    pred = model(im, augment=augment, visualize=visualize)
    t3 = time_sync()
    dt[1] += t3 - t2

    # NMS
    pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
    dt[2] += time_sync() - t3
    
    # Process predictions
    for i, det in enumerate(pred):  # per image
        seen += 1
        construct_machines = {}
        person_detect = {}

        p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)
        
        ratio = input_ratio(height,im0)
        p = Path(p)  # to Path
        save_path = str(save_dir / p.name)  # im.jpg
        txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
        s += '%gx%g ' % im.shape[2:]  # print string
        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        imc = im0.copy() if save_crop else im0  # for save_crop
        #im0 = np.uint8(im0 @ matrix_cam2world)
        annotator = Annotator(im0, line_width=line_thickness, example=str(names))
        
        if len(det):
            
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()
            

            # Print results
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()  # detections per class
                
                s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                
            # Write results
            for *xyxy, conf, cls in reversed(det):
                
                if save_txt:  # Write to file
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    
                    line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                    if int(cls)==18 or int(cls)==19:
                        construct_machines[int(cls)] = xywh[:2]
                        
                        
                    elif int(cls) in (0,1,2,4):
                        person_detect[ int(cls)] = xywh[:2]
                        
                        
                    with open(f'{txt_path}.txt', 'a') as f:
                        f.write(('%g ' * len(line)).rstrip() % line + '\n')

                if save_img or save_crop or view_img:  # Add bbox to image
                    c = int(cls)  # integer class
                    label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                    
                    annotator.box_label(xyxy, label, color=colors(c, True))
                    
                if save_crop:
                    save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)
                
                #Draw Distance Line & Show Distance between objects
                for position in person_detect.values():
                    
                    x1,y1 = position
                    x1center = (x1 * im0.shape[1])
                    y1center = (y1  * im0.shape[0])

                    for position2 in construct_machines.values():
                        x2,y2 = position2
                        x2center = (x2 * im0.shape[1])
                        y2center = (y2 * im0.shape[0])
                        start_point = (int(x1center),int(y1center))
                        end_point = (int(x2center),int(y2center))
                        D = Euc_dist(x1center,x2center,y1center,y2center)
                        real_d = ratio*D
                        (mX, mY) = (((x2center+x1center)/2), ((y2center+y1center)/2))
                        if real_d <= threshold_dist:
                            cv2.line(im0, start_point, end_point,(0,0,255),2)
                            cv2.putText(im0, "{:.1f}m".format(real_d), (int(mX), int(mY - 10)),cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                            
            # Stream results
            im0 = annotator.result()
            if view_img:
                if p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond
            
            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                    #cv2.imwrite(save_path,img2)
                    
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)
        # Print time (inference-only)
        LOGGER.info(f'{s}Done. ({t3 - t2:.3f}s)')
    
    # Print results
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    