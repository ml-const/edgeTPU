import time

import cv2
import os
import sys
#import torch
import tflite_runtime
#from tensorflow.python.eager import context
#_ = tf.Variable([8])

#context._context = None
#context._create_context()

#tf.config.threading.set_inter_op_parallelism_threads(8)
#tf.config.set_soft_device_placement(True)

import numpy as np
import random

def clip_coords(boxes, img_shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    #boxes[:, 0].clamp_(0, img_shape[1])  # x1
    #boxes[:, 1].clamp_(0, img_shape[0])  # y1
    #boxes[:, 2].clamp_(0, img_shape[1])  # x2
    #boxes[:, 3].clamp_(0, img_shape[0])  # y2
    boxes[:,0] = np.clip(boxes[:,0], 0, img_shape[1])# x1
    boxes[:,1] = np.clip(boxes[:,1], 0, img_shape[0])# x1
    boxes[:,2] = np.clip(boxes[:,2], 0, img_shape[1])# x1
    boxes[:,3] = np.clip(boxes[:,3], 0, img_shape[0])# x1

def scale_coords(img0_shape, coords, img1_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    #print(coords)
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]
        
    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    #print(coords)
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


def detect(source,weights,imgsz,trace=False):
    start = time.time()
    #conf_thres = 0.25
    #iou_thres = 0.45
    #max_output_size = 100
    # Directories
    names = ["tank","armored vehicles","artillery","air defense",
             "MLRS","cargo vehicles", "tankers","aircrafts","helicopters"]
    #colors = {name:[random.randint(0, 255) for _ in range(3)] for i,name in enumerate(names)}
    
       
    # Load model
    #interpreterOptions = tf.lite.Interpreter.Options()
    #interpreterOptions.setUseXNNPACK(True)
    interpreter = tflite_runtime.Interpreter(model_path=weights, num_threads=4)

    # Get input and output tensors
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.allocate_tensors()
    
    video = cv2.VideoCapture(source)
    #data =  os.path.basename(source)
    #name = os.path.splitext(data)[0]
    #amoun_frs = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if not video.isOpened():
      print("Could not open video")
      sys.exit()
      
    ok, frame = video.read()
    
    if not ok:
      print ('Cannot read video file')
      sys.exit()  
     
    #h,w,c = frame.shape       
   # vid_writer = cv2.VideoWriter(name+'_tfLite_ft16'+'.mp4', 
                                # cv2.VideoWriter_fourcc(*'mp4v'), 30, (w, h))
    
    while True:
        #t0 = time.time()
        # Read a new frame
        #start_time = time.time()
        ok, frame = video.read()
        if not ok:
            break
             
            # Start timer
        timer = cv2.getTickCount()
        
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        np_features, ratio, dwdh = letterbox(frame, auto=False)
        np_features = np.array(np_features/255, dtype=np.float32)
        np_features = np_features.transpose((2, 0, 1))
        np_features = np.expand_dims(np_features, axis=0)
        np_features = np.ascontiguousarray(np_features)
        
       #print(np_features.shape)
        #t2 = time.time()
        interpreter.set_tensor(input_details[0]['index'], np_features)
        interpreter.invoke()

        output_data = interpreter.get_tensor(output_details[0]['index'])
        #output_data2= np.delete(output_data, 0, 1)
        #output_data2 = output_data2[:,[0,1,2,3,5,4]]
        
        #pred_2 = tf.image.non_max_suppression(
        #        output_data[:,1:5],
        #        output_data[:,5],
        #        max_output_size,
        #        iou_threshold=iou_thres,
        #        score_threshold=float('-inf'),
        #        name=None
        #        )
        
        #img_res = frame.copy()
        if output_data.any():
            output_data[:,1:5] = scale_coords((frame.shape)[:2],
                                    output_data[:,1:5],(np_features.shape)[2:4]).round()
       #img_res = cv2.resize(img_res,(640,640))
        #print(output_data2[pred_2,:4])
            fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
            #t1 = time.time()
            #print(f'Time of detection with reading frame = {(t1-t0):.4f}s\n Time of detection = {(t1-t2):.4f}s')
            for coordinate in output_data[:,:]:
                #print(coordinate)
                xyxy = [int(coordinate[1]),int(coordinate[2]),int(coordinate[3]),int(coordinate[4])]
                label_ind = int(coordinate[5])
                label = names[label_ind]
                
                print(f'Detected fps= {fps}')
                print(f'\nBBox:[{xyxy}]')
                print(f'\nTarget: {label}')
            
                
        else:
            fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
            print(f'NOTDetected fps= {fps}')

        #cv2.imshow("Tracking", img_res)
        #cv2.waitKey(1)
        #filename='KCF/KCF_{0}.jpg'.format(ii)
        #cv2.imwrite(filename, frame)
        #vid_writer.write(img_res)    
        
    print(f'Done. ({time.time() - start:.3f}s)')
    video.release()
    #vid_writer.release()        
    #cv2.destroyWindow('Tracking')   
    return output_data


if __name__ == '__main__':
    
    weights = 'D:/yolov7/PyTorch-ONNX-TFLite-master/conversion/model_v5_ft16.tflite'
    source = 'D:/Tracker_trying/Tracker_video/movie_012_.mp4'
    imgsz = 640
    #tf.config.threading.set_inter_op_parallelism_threads(num_threads=8) 
    #tf.config.threading.set_intra_op_parallelism_threads()
    #tf.config.set_soft_device_placement(enabled)
    
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #print(f'Device are found: {device.type}')
    #check_requirements(exclude=('pycocotools', 'thop'))

    #with torch.no_grad():
    coord = detect(source,weights,imgsz,trace=False)