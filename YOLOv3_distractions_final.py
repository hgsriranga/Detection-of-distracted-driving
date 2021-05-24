import cv2
import random
import glob
import numpy as np

# Insert necessary files of YOLOv3
net = cv2.dnn.readNet("yolov3_custom_last_4000.weights", "yolov3_custom_final.cfg")


# List of classes containing various driving distractions
classes_DD = []
with open("obj.names", "r") as o:
    classes_DD=[line.strip() for line in o.readlines()]

# Insert the path of images
images_path = glob.glob(r"C:\Users\Lenovo\Desktop\CHALMERS COURSES\Thesis\secondary task analysis\database\v2_cam1_cam2_ split_by_driver\Camera 2\test\c0\*.jpg")

# Responsible for getting final detections
layer_names = net.getLayerNames()
final_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
colours = np.random.uniform(0, 255, size=(len(classes_DD), 3))

# Insert the path of images
random.shuffle(images_path)
# To run a loop through all the images
for img_path in images_path:
    # Loading image
    img = cv2.imread(img_path)
    img = cv2.resize(img, None, fx=1.0, fy=1.0)
    height, width, channels = img.shape

    # predicting objects
    blob_DD = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

    net.setInput(blob_DD)
    final = net.forward(final_layers)

    # Showing information on the screen
    class_ids = []
    confidences_score = []
    bounding_boxes = []
    for fin in final:
        for prediction in fin:
            scores_box = prediction[5:]
            class_id = np.argmax(scores_box)
            confidence_box = scores_box[class_id]
            if confidence_box > 0.5:
                # bounding box prediction for distractions
                print(class_id)
                pos_x = int(prediction[0] * width)
                pos_y = int(prediction[1] * height)
                w = int(prediction[2] * width)
                h = int(prediction[3] * height)

                # Rectangle coordinates
                x = int(pos_x - w / 2)
                y = int(pos_y - h / 2)

                bounding_boxes.append([x, y, w, h])
                confidences_score.append(float(confidence_box))
                class_ids.append(class_id)

    bounding_box = cv2.dnn.NMSBoxes(bounding_boxes, confidences_score, 0.5, 0.4)
    print(bounding_box)
    font = cv2.FONT_HERSHEY_SCRIPT_COMPLEX
    for i in range(len(bounding_box)):
        if i in bounding_boxes:
            x, y, w, h = bounding_boxes[i]
            label = str(classes_DD[class_ids[i]])
            colour = colours[class_ids[i]]
            cv2.rectangle(img, (x, y), (x + w, y + h), colour, 2)
            cv2.putText(img, label, (x, y + 30), font, 3, colour, 2)


    cv2.imshow("Distraction", img)
    #small = cv2.resize(img, (0, 0), fx=0.4, fy=0.4)
    #cv2.imshow("small",img)
    key = cv2.waitKey(0)
    cv2.destroyAllWindows()