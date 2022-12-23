import time

import cv2
import os
import sys
from pycoral.utils.edgetpu import make_interpreter
from pycoral.adapters import common
from pycoral.adapters import detect

import numpy as np
import random

def clip_coords(boxes, img_shape):
    boxes[:,0] = np.clip(boxes[:,0], 0, img_shape[1])# x1
    boxes[:,1] = np.clip(boxes[:,1], 0, img_shape[0])# x1
    boxes[:,2] = np.clip(boxes[:,2], 0, img_shape[1])# x1
    boxes[:,3] = np.clip(boxes[:,3], 0, img_shape[0])# x1

def scale_coords(img0_shape, coords, img1_shape, ratio_pad=None):
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]
        
    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)
    return coords

def plot_one_box(x, img, color=None, label=None, line_thickness=3, fps = None):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
    if fps:
        cv2.putText(img, "FPS : " + str(int(fps)), (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,0,0), 1)

def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, r, (dw, dh)


def detection(source,weights,imgsz,trace=False):
    start = time.time()
    conf_thres = 0.25

    names = ["tank","armored vehicles","artillery","air defense",
             "MLRS","cargo vehicles", "tankers","aircrafts","helicopters"]

    interpreter = make_interpreter(weights)
    interpreter.allocate_tensors()
    size = common.input_size(interpreter)
    
    video = cv2.VideoCapture(source)
    data =  os.path.basename(source)
    name = os.path.splitext(data)[0]

    
    if not video.isOpened():
      print("Could not open video")
      sys.exit()
      
    ok, frame = video.read()
    
    if not ok:
      print ('Cannot read video file')
      sys.exit()  
     
    h,w,c = frame.shape       
    vid_writer = cv2.VideoWriter(name+'_tfLite_ft16'+'.mp4', 
                                 cv2.VideoWriter_fourcc(*'mp4v'), 30, (w, h))
    
    while True:
        ok, frame = video.read()
        if not ok:
            break
             
            # Start timer
        timer = cv2.getTickCount()
        
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        np_features, ratio, dwdh = letterbox(frame ,new_shape=(size,size), auto=False)
        np_features = np.array(np_features/255, dtype=np.float32)
        np_features = np_features.transpose((2, 0, 1))
        np_features = np.expand_dims(np_features, axis=0)
        np_features = np.ascontiguousarray(np_features)
        
       #print(np_features.shape)
        t2 = time.time()
        common.set_input(interpreter, frame)
        interpreter.invoke()
        objs = detect.get_objects(interpreter, conf_thres, dwdh)
        t3 = time.time()
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
        
        img_res = frame.copy()
        if not objs:
            print('No objects detected')
            print(f'Time for detecting: {t3 - t2:.3f}s' )
            print(f'FPS = {fps}')
        
        for obj in objs:
            print(names.get(obj.id, obj.id))
            print('  id:    ', obj.id)
            print('  score: ', obj.score)
            print('  bbox:  ', obj.bbox)
            print(f'Time for detecting: {t3 - t2:.3f}s' )
            print(f'FPS = {fps}')
      
        vid_writer.write(img_res)    
        
    print(f'Done. ({time.time() - start:.3f}s)')
    video.release()
    
    return obj


if __name__ == '__main__':
    
    weights = './model_v5_int8.tflite'
    source = './movie_012_.mp4'
    imgsz = 640
    
    coord = detection(source,weights,imgsz,trace=False)