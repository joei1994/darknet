from ctypes import *
import math
import random
import os
import cv2
import numpy as np
import time
import darknet
import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
from sort import *
import time
from pdb import set_trace

iou_threshold = .6
current_milli_time = lambda: int(round(time.time() * 1000))
previous_taken_time = current_milli_time()

def bbox_area(xmin, ymin, xmax, ymax):
    diff_x = xmax - xmin
    diff_y = ymax - ymin

    return diff_x * diff_y

def convertBack(x, y, w, h):
    xmin = int(round(x - (w / 2)))
    xmax = int(round(x + (w / 2)))
    ymin = int(round(y - (h / 2)))
    ymax = int(round(y + (h / 2)))
    return xmin, ymin, xmax, ymax

def createDetection(xmin, ymin, xmax, ymax, confidence, label):
    det = {}
    det['xmin'] = xmin
    det['ymin'] = ymin
    det['xmax'] = xmax
    det['ymax'] = ymax
    det['confidence'] = confidence
    det['label'] = label

    return  det

def extractSingleDetection(detection):
    x, y, w, h = detection[2][0],\
            detection[2][1],\
            detection[2][2],\
            detection[2][3]
    xmin, ymin, xmax, ymax = convertBack(float(x), float(y), float(w), float(h))

    return  createDetection(xmin, ymin, xmax, ymax, detection[1], detection[0].decode())
    
def extractDetections(detections):
    return [extractSingleDetection(det) for det in detections]  

def is_inside(target_det, detections):
    for det in detections:
        if target_det['xmin'] > det['xmin'] & target_det['ymin'] > det['ymin'] & target_det['xmax'] < det['xmax'] & target_det['ymax'] < det['ymax']:
            return True
    return False 

def filterDetections(detections, min_area):
    # Filter out detections which are not car
    detections = [det for det in detections if (det['label'].lower() == 'car') or (det['label'].lower() == 'truck')]

    # Filter out detections which are too small
    detections = [det for det in detections if bbox_area(det['xmin'], det['ymin'], det['xmax'],det['ymax']) > min_area]

    # Filter out detection which inside others
    def filter_inside(detections):
        indices_to_remove = set()
        for i, detection in enumerate(detections):
            if is_inside(detection, detections):
                indices_to_remove.add(i)
                
        all_indices = set(range(len(detections)))
        remain_indices = all_indices - indices_to_remove
        return  [det for i, det in enumerate(detections) if i in remain_indices]  

    # detections = filter_inside(detections)

    return detections    

def resize_frame_and_bbox(frame, detections, weight, height):
    bboxes = BoundingBoxesOnImage(
        [BoundingBox(x1=det['xmin'],y1=det['ymin'],x2=det['xmax'],y2=det['ymax']) for det in detections],
        shape=frame.shape
    )

    resize = iaa.Resize({"height": height, "width": weight})
    resized_frame, bboxes = resize(image=frame, bounding_boxes=bboxes)

    return resized_frame, [createDetection(int(bbox.x1), int(bbox.y1), int(bbox.x2), int(bbox.y2), detections[i]['confidence'], detections[i]['label']) for i, bbox in enumerate(bboxes.bounding_boxes)]

def assign_ids(detections, tracker):
    dets = np.asarray([[det['xmin'], det['ymin'], det['xmax'], det['ymax'], det['confidence']] for det in detections])
    tracks = tracker.update(dets)

    return tracks

def drawBoxesWithId(tracks, img):
    for track in tracks:
        xmin, ymin, xmax, ymax = track[0], track[1], track[2], track[3]
        object_id = track[4]
        
        pt1 = (xmin, ymin)
        pt2 = (xmax, ymax)

        cv2.rectangle(img, pt1, pt2, (0, 255, 0), 1)
        
        cv2.putText(img, f"{object_id}", (pt1[0], pt1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, [0, 255, 0], 2)
    return img

def box_iou(b1_xmin, b1_ymin, b1_xmax, b1_ymax, b2_xmin, b2_ymin, b2_xmax, b2_ymax):
    intersect_xmin = max(b1_xmin, b2_xmin)
    intersect_ymin = max(b1_ymin, b2_ymin)
    intersect_xmax = min(b1_xmax, b2_xmax)
    intersect_ymax = min(b1_ymax, b2_ymax)

    intersect_area = max(0, intersect_xmax - intersect_xmin + 1) * max(0, intersect_ymax - intersect_ymin + 1)

    box1_area = (b1_xmax - b1_xmin + 1) * (b1_ymax - b1_ymin + 1)
    box2_area = (b2_xmax - b2_xmin + 1) * (b2_ymax - b2_ymin + 1)

    total_area = box1_area + box2_area - intersect_area
    return intersect_area  / total_area

def converColor(frame):
    try:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    except:
        return frame

def capture(tracks, roi_box, frame, captured_ids, output_dir='./capture'):
    global previous_taken_time
    margin = 30

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for track in tracks:
        if box_iou(track[0], track[1], track[2], track[3], roi_box['xmin'], roi_box['ymin'], roi_box['xmax'], roi_box['ymax']) > iou_threshold:
            
            object_id = track[4]
            if object_id not in captured_ids:
                captured_ids.add(object_id)
                cropped_frame = frame[track[1] - margin:track[3] + margin, track[0] - margin:track[2] + margin]
                cropped_frame = converColor(cropped_frame)

                output_path = os.path.join(output_dir, f'{current_milli_time()}.jpg')
                if not os.path.exists(output_path):
                    current_taken_time = current_milli_time()
                    time_diff = current_taken_time - previous_taken_time
                    #print(f"{current_taken_time}, {object_id}")
                    if time_diff > 1000:
                        cv2.imwrite(output_path, cropped_frame)
                    previous_taken_time = current_taken_time
    
    return captured_ids

def create_roi_box(x, y, width, height):
    roi_box = {
        'xmin': int(x - width / 2),
        'ymin': int(y - height / 2),
        'xmax': int(x + width / 2),
        'ymax': int(y + height / 2)
    }

    return roi_box

netMain = None
metaMain = None
altNames = None

def YOLO():
    captured_ids = set()

    global metaMain, netMain, altNames
    configPath = "./cfg/yolov3-tiny.cfg"
    weightPath = "./yolov3-tiny.weights"
    metaPath = "./cfg/coco.data"
    if not os.path.exists(configPath):
        raise ValueError("Invalid config path `" +
                         os.path.abspath(configPath)+"`")
    if not os.path.exists(weightPath):
        raise ValueError("Invalid weight path `" +
                         os.path.abspath(weightPath)+"`")
    if not os.path.exists(metaPath):
        raise ValueError("Invalid data file path `" +
                         os.path.abspath(metaPath)+"`")
    if netMain is None:
        netMain = darknet.load_net_custom(configPath.encode(
            "ascii"), weightPath.encode("ascii"), 0, 1)  # batch size = 1
    if metaMain is None:
        metaMain = darknet.load_meta(metaPath.encode("ascii"))
    if altNames is None:
        try:
            with open(metaPath) as metaFH:
                metaContents = metaFH.read()
                import re
                match = re.search("names *= *(.*)$", metaContents,
                                  re.IGNORECASE | re.MULTILINE)
                if match:
                    result = match.group(1)
                else:
                    result = None
                try:
                    if os.path.exists(result):
                        with open(result) as namesFH:
                            namesList = namesFH.read().strip().split("\n")
                            altNames = [x.strip() for x in namesList]
                except TypeError:
                    pass
        except Exception:
            pass

    frame_width, frame_height = 1920, 1080    
    #roi_box = create_roi_box(x=680, y=520, width=560, height=310)
    roi_box = create_roi_box(x=680, y=570, width=640, height=400)    
    cap = cv2.VideoCapture('rtsp://admin:iapp2019@192.168.1.64/1')
    #cap = cv2.VideoCapture("data/cars.mp4")
    cap.set(3, frame_width)
    cap.set(4, frame_height)

    #out = cv2.VideoWriter("output.avi", cv2.VideoWriter_fourcc(*"MJPG"), 10, (frame_width, frame_height))

    # Create an image we reuse for each detect
    darknet_image = darknet.make_image(darknet.network_width(netMain),
                                    darknet.network_height(netMain),3)
    # Create object Tracker
    tracker = Sort()

    while True:
        prev_time = time.time()
        _, frame = cap.read()
        original_frame = converColor(frame)
        # out.write(original_frame)

        try:
            resized_frame = cv2.resize(original_frame,
                                    (darknet.network_width(netMain),
                                        darknet.network_height(netMain)),
                                    interpolation=cv2.INTER_LINEAR)
        except:
            continue

        darknet.copy_image_from_bytes(darknet_image,resized_frame.tobytes())

        # detection on resized frame
        detections = darknet.detect_image(netMain, metaMain, darknet_image, thresh=.25)
        detections = extractDetections(detections)

        # Resize frame and bbox
        _, detections = resize_frame_and_bbox(resized_frame, detections, frame_width, frame_height)

        # Filter out some detections
        roi_box_area = bbox_area(roi_box['xmin'], roi_box['ymin'], roi_box['xmax'], roi_box['ymax'])
        detections = filterDetections(detections, min_area = (.3) * roi_box_area)

        # Assign IDs to each detections
        tracks = assign_ids(detections, tracker)
        tracks = np.asarray(tracks, dtype='int32')

        # Capture 
        captured_ids = capture(tracks, roi_box, original_frame, captured_ids)
        captured_ids = set([track[4] for track in tracks]).intersection(captured_ids)

        # Draw bbox on image
        frame = drawBoxesWithId(tracks, original_frame)

        # Draw ROI box
        cv2.rectangle(frame, (roi_box['xmin'], roi_box['ymin']), (roi_box['xmax'], roi_box['ymax']), (255, 0, 0), 1)
        
        frame = converColor(frame)
        cv2.imshow('Demo', frame)
        #print(1/(time.time()-prev_time + .0001))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    out.release()

if __name__ == "__main__":
    YOLO()
