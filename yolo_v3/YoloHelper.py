import cv2
import numpy as np

# 
# ref: https://pjreddie.com/darknet/yolo/ https://github.com/pjreddie/darknet
# https://github.com/pjreddie/darknet/blob/master/LICENSE
#
# For this to work pl download following files
#   https://pjreddie.com/media/files/yolov3.weights
#   https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg
#   https://github.com/pjreddie/darknet/blob/master/data/coco.names
#

def load_yolo(weight_file, cfg_file, coco_file):
    print('Loading yolo..')
    net = cv2.dnn.readNet(weight_file, cfg_file)
    #save all the names in file o the list classes
    classes = []
    with open(coco_file, "r") as f:
        classes = [line.strip() for line in f.readlines()]
    #get layers of the network
    layer_names = net.getLayerNames()
    #Determine the output layer names from the YOLO model 
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    print('Yolo loaded!')
    return net, output_layers, classes

def detect_obj(yolo_net, yolo_output_layers, yolo_classes, img, 
               score_threshold=0.5, nms_threshold=0.4, only=None):
    #img = cv2.resize(img, None, fx=0.4, fy=0.4)
    box_img = img.copy()
    height, width, channels = img.shape

    # USing blob function of opencv to preprocess image
    blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    #Detecting objects
    yolo_net.setInput(blob)
    outs = yolo_net.forward(yolo_output_layers)

    # Showing informations on the screen
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > score_threshold:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    #We use NMS function in opencv to perform Non-maximum Suppression
    #we give it score threshold and nms threshold as arguments.
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold, nms_threshold)
    colors = np.random.uniform(0, 255, size=(len(yolo_classes), 3))
    outputs = []
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(yolo_classes[class_ids[i]])
            if only is not None and (label in only)==False:
                continue
            color = colors[class_ids[i]]
            cv2.rectangle(box_img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(box_img, label, (x, y -5),cv2.FONT_HERSHEY_SIMPLEX,1/2, color, 2)
            outputs.append([label, (x,y,w,h), round(confidences[i],2)])

    #outputs = [[label,(x,y,w,h),confidences],..]        
    return outputs, box_img

def crop_output(src_img, output, pad=0):
    (x, y, w, h) = output[1]
    y_pad = y-pad if y-pad > 0 else y
    x_pad = x-pad if x-pad > 0 else x
    img_crop = src_img[y_pad:y+h+pad, x_pad:x+w+pad]
    return img_crop

def make_image_square(img, color=(255,255,255)):
    large_side = img.shape[0] if img.shape[0] > img.shape[1] else img.shape[1]
    old_image_height, old_image_width, channels = img.shape
    result = np.full((large_side,large_side, channels), color, dtype=np.uint8)

    # compute center offset
    x_center = (large_side - old_image_width) // 2
    y_center = (large_side - old_image_height) // 2

    # copy img image into center of result image
    result[y_center:y_center+old_image_height, 
           x_center:x_center+old_image_width] = img
    return result

if __name__ == '__main__':
    print('Testing yolo..')
    yolo_net, yolo_output_layers, yolo_classes = YoloHelper.load_yolo('yolov3.weights','yolov3.cfg', 'coco.names')
    img = cv2.imread("test_img.jpg")
    outputs, img = YoloHelper.detect_obj(yolo_net, yolo_output_layers, yolo_classes, img)

    #label,(x,y,w,h),confidences
    for output in outputs:
        print(output)