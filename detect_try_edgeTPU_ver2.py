import time

import cv2
import os
import sys
from pycoral.utils.edgetpu import make_interpreter
from pycoral.adapters import common
#from pycoral.adapters import detect
from functions import non_max_suppression, get_image_tensor, get_scaled_coords
import numpy as np




def detection(source,weights,imgsz,trace=False):
    start = time.time()
    #conf_thres = 0.25
    mean = 128 #'Mean value for input normalization'
    std = 128 #'STD value for input normalization'
    names = ["tank","armored vehicles","artillery","air defense",
             "MLRS","cargo vehicles", "tankers","aircrafts","helicopters"]

    interpreter = make_interpreter(weights)
    interpreter.allocate_tensors()
    params = common.input_details(interpreter, 'quantization_parameters')
    input_scale = params['scales']
    input_zero = params['zero_points']
    output_details = interpreter.get_output_details()[0]
    output_scale, output_zero = output_details['quantization']
    video = cv2.VideoCapture(source)
    data =  os.path.basename(source)
    name = os.path.splitext(data)[0]
    input_shape = common.input_size(interpreter)
    
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
        full_image, np_features1, pad = get_image_tensor(frame, max(input_shape))
        np_features =(np_features1/input_scale) + input_zero
        np_features = np_features[np.newaxis].astype(np.uint8)
        
       #print(np_features.shape)
        t2 = time.time()
        #common.set_input(interpreter, frame)
        if abs(input_scale * std - 1) < 1e-5 and abs(mean - input_zero) < 1e-5:
            # Input data does not require preprocessing.
            common.set_input(interpreter, frame)
        else:
            # Input data requires preprocessing
            normalized_input = (np.asarray(frame) - mean) / (std * input_scale) + input_zero
            np.clip(normalized_input, 0, 255, out=normalized_input)
            common.set_input(interpreter, normalized_input.astype(np.uint8))
        interpreter.invoke()
        result = ((interpreter.tensor(interpreter.get_output_details()[0]['index'])()).astype('float32') - output_zero) * output_scale
        new_data = non_max_suppression(result)
        coord = new_data[0]
        coord = get_scaled_coords(xyxy=coord[:,:4], output_image=frame, pad=pad)
        t3 = time.time()
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
        
        img_res = frame.copy()
        if not len(coord):
            print('No objects detected')
            print(f'Time for detecting: {t3 - t2:.3f}s' )
            print(f'FPS = {fps}')
            cv2.putText(img_res, "FPS : " + str(int(fps)), (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,0,0), 1)
            cv2.putText(img_res, 'Target not found', (10,40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,255), 1)
        else:
            for ind in range(0,len(coord)):
                obj_id = int(new_data[0][ind][5])
                label = names[obj_id]
                scores = new_data[0][ind][4]
                print(label)
                #print('  id:    ', obj_id)
                #print('  score: ', scores)
                #print('  bbox:  ', coord[ind][:])
                print(f'Time for detecting: {t3 - t2:.3f}s' )
                print(f'FPS = {fps}')
                p1 = (int(coord[ind][0]), int(coord[ind][1]))
                p2 = (int(coord[ind][2])), int(coord[ind][3])
                cv2.rectangle(img_res, p1, p2, (255,0,0), 2, 1)
                cv2.putText(img_res, "FPS : " + str(int(fps)), (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,0,0), 1)
                cv2.putText(img_res, str(label)+str(scores), (int(coord[ind][0])+2, int(coord[ind][1])+2), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,0,0), 1)
  
        vid_writer.write(img_res)    
    
    print(f'Done. ({time.time() - start:.3f}s)')
    video.release()
    vid_writer.release()

    return coord


if __name__ == '__main__':
    
    weights = './5nano_seg-int8_edgetpu.tflite'
    source = './movie_012_.mp4'
    imgsz = 640
    
    coord = detection(source,weights,imgsz,trace=False)
