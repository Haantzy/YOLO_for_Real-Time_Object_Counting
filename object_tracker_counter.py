import time, random
import numpy as np
from absl import app, flags, logging
from absl.flags import FLAGS
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from yolov3_tf2.models import (YoloV3, YoloV3Tiny)
from yolov3_tf2.dataset import transform_images
from yolov3_tf2.utils import draw_outputs, convert_boxes
from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
from tools.object_history import hist
from tools.Intersections import create_intersection, intersection, draw_intersections
import csv
import os
from PIL import Image
import copy

from tqdm import tqdm
from cv2 import CAP_PROP_FRAME_COUNT

flags.DEFINE_string('classes', './data/labels/coco.names', 'path to classes file')
flags.DEFINE_string('weights', './weights/yolov3.tf',
                    'path to weights file')
flags.DEFINE_boolean('tiny', False, 'yolov3 or yolov3-tiny')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_integer('num_classes', 80, 'number of classes in the model')

flags.DEFINE_string('video', './data/video/test.mp4',
                    'path to video file or number for webcam)')
flags.DEFINE_string('output_format', 'XVID', 'codec used in VideoWriter when saving video to file')

def main(_argv):
    #Create the Output Files and their respective folder
    intersection_dir = './data/Results_count'
    try:
        os.mkdir(intersection_dir)
    except FileExistsError:
        pass

    result_vid_dir = './data/Results_vid'
    try:
        os.mkdir(result_vid_dir)
    except FileExistsError:
        pass

    base_name = os.path.splitext(os.path.basename(FLAGS.video))[0]
    intersection_file = intersection_dir + '/' + base_name + '_Results.csv'
    video_file = result_vid_dir + '/' + base_name + '_Results.avi'

    #Open the intersection writer
    f = open(intersection_file,'w')
    writer = csv.writer(f)

    #Create the categories for the intersection file
    header = ['intersection_num','Frame_num','car_increase']
    writer.writerow(header)

    # Definition of the parameters
    max_cosine_distance = 0.5
    nn_budget = None
    nms_max_overlap = 1.0
    
    #initialize deep sort
    model_filename = 'model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)

    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

    if FLAGS.tiny:
        yolo = YoloV3Tiny(classes=FLAGS.num_classes)
    else:
        yolo = YoloV3(classes=FLAGS.num_classes)

    yolo.load_weights(FLAGS.weights)
    logging.info('weights loaded')

    class_names = [c.strip() for c in open(FLAGS.classes).readlines()]
    logging.info('classes loaded')

    try:
        vid = cv2.VideoCapture(int(FLAGS.video))
    except:
        vid = cv2.VideoCapture(FLAGS.video)

    out = None

    # by default VideoCapture returns float instead of int
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(vid.get(cv2.CAP_PROP_FPS))
    codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)
    out = cv2.VideoWriter(video_file, codec, fps, (width, height))
    list_file = open('detection.txt', 'w')
    frame_index = -1
    
    fps = 0.0
    count = 0

    #Dictionary of object and bbox history
    frame_num = 0
    tracked_objects = {}

    #Create a Dictionary of intersection Crossings
    intersections = {}

    #Create an allowed list of classes
    class_allowed = ['bicycle','car','motorbike','bus','truck','boat']

    #Show_dims for onscreen window
    x_show = 1280
    y_show = 720
    
    while True:
        frame_num += 1
        _, img = vid.read()

        #Some videos look normal but enter flipped, use this to correct
        img = cv2.flip(img,0)
        img = cv2.flip(img,1)

        if img is None:
            logging.warning("Empty Frame")
            time.sleep(0.1)
            count+=1
            if count < 3:
                continue
            else: 
                break

        img_in = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
        img_in = tf.expand_dims(img_in, 0)
        img_in = transform_images(img_in, FLAGS.size)

        # Define the intersections
        if frame_num == 1:
            img_small = cv2.resize(img, (x_show, y_show))
            params = [intersections, img,x_show,y_show]
            cv2.imshow('output', img_small)

            print("")
            print("Press and hold Left click to start a line, release to end")
            print("Press right click to remove last line")
            print("Press Q to end the intersection design")

            while True:
                cv2.setMouseCallback('output', create_intersection,params)

                if cv2.waitKey(0) & 0xFF == ord('q'):
                    break

            #Display the intersections
            print("Number of Intersections: " + str(len(intersections)))

        t1 = time.time()
        boxes, scores, classes, nums = yolo.predict(img_in)
        classes = classes[0]
        names = []
        for i in range(len(classes)):
            names.append(class_names[int(classes[i])])
        names = np.array(names)
        converted_boxes = convert_boxes(img, boxes[0])
        features = encoder(img, converted_boxes)

        detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in zip(converted_boxes, scores[0], names, features) if class_name in class_allowed]

        
        #initialize color map
        cmap = plt.get_cmap('tab20b')
        colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

        # run non-maxima suppresion
        boxs = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        classes = np.array([d.class_name for d in detections])
        indices = preprocessing.non_max_suppression(boxs, classes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]        

        # Call the tracker
        tracker.predict()
        tracker.update(detections)

        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox = track.to_tlbr()
            class_name = track.get_class()
            color = colors[int(track.track_id) % len(colors)]
            color = [i * 255 for i in color]
            cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
            cv2.rectangle(img, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(len(class_name))*17, int(bbox[1])), color, -1)
            cv2.putText(img, class_name,(int(bbox[0]), int(bbox[1]-10)),0, 0.75, (255,255,255),2)

            #Dictionary of object history
            if track.track_id in tracked_objects:
                tracked_objects[track.track_id].add(track,frame_num)
            else:
                tracked_objects[track.track_id] = hist(track,frame_num,color)

            #Draw the car history
            #tracked_objects[track.track_id].draw_last_x(100,img)

            #Check for cars crossing the line
            for key in intersections:
                intersections[key].check_for_crossing(tracked_objects[track.track_id],frame_num)

        draw_intersections(intersections,img)

        ### UNCOMMENT BELOW IF YOU WANT CONSTANTLY CHANGING YOLO DETECTIONS TO BE SHOWN ON SCREEN
        #for det in detections:
        #    bbox = det.to_tlbr() 
        #    cv2.rectangle(img,(int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(255,0,0), 2)
        
        # print fps on screen 
        fps  = ( fps + (1./(time.time()-t1)) ) / 2
        cv2.putText(img, "FPS: {:.2f}".format(fps), (0, 30),
                          cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)

        out.write(img)
        frame_index = frame_index + 1
        list_file.write(str(frame_index)+' ')
        if len(converted_boxes) != 0:
            for i in range(0,len(converted_boxes)):
                list_file.write(str(converted_boxes[i][0]) + ' '+str(converted_boxes[i][1]) + ' '+str(converted_boxes[i][2]) + ' '+str(converted_boxes[i][3]) + ' ')
        list_file.write('\n')

        #Show the image
        img_small = cv2.resize(img, (x_show, y_show))
        cv2.imshow('output', img_small)

        #Write the intersection data
        for i in intersections:
            writer.writerow([i,frame_num,intersections[i].out()])

        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

    vid.release()
    f.close()

    out.release()
    list_file.close()


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
