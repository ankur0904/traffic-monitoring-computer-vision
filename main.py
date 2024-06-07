import numpy as np
from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import *

# cap = cv2.VideoCapture(0)  # webcam
# cap.set(3, 1280)  # width
# cap.set(4, 720)  # height

cap = cv2.VideoCapture("1.mp4")  # video file
model = YOLO("yolov8n.pt")
mask = cv2.imread('mask2.png')  # depends on your input video stream

# tracker
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

# line, when car bypass the line increase the count
line = [500, 800, 950, 800]

classNames = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    4: "airplane",
    5: "bus",
    6: "train",
    7: "truck",
    8: "boat",
    9: "traffic light",
    10: "fire hydrant",
    11: "stop sign",
    12: "parking meter",
    13: "bench",
    14: "bird",
    15: "cat",
    16: "dog",
    17: "horse",
    18: "sheep",
    19: "cow",
    20: "elephant",
    21: "bear",
    22: "zebra",
    23: "giraffe",
    24: "backpack",
    25: "umbrella",
    26: "handbag",
    27: "tie",
    28: "suitcase",
    29: "frisbee",
    30: "skis",
    31: "snowboard",
    32: "sports ball",
    33: "kite",
    34: "baseball bat",
    35: "baseball glove",
    36: "skateboard",
    37: "surfboard",
    38: "tennis racket",
    39: "bottle",
    40: "wine glass",
    41: "cup",
    42: "fork",
    43: "knife",
    44: "spoon",
    45: "bowl",
    46: "banana",
    47: "apple",
    48: "sandwich",
    49: "orange",
    50: "broccoli",
    51: "carrot",
    52: "hot dog",
    53: "pizza",
    54: "donut",
    55: "cake",
    56: "chair",
    57: "couch",
    58: "potted plant",
    59: "bed",
    60: "dining table",
    61: "toilet",
    62: "tv",
    63: "laptop",
    64: "mouse",
    65: "remote",
    66: "keyboard",
    67: "cell phone",
    68: "microwave",
    69: "oven",
    70: "toaster",
    71: "sink",
    72: "refrigerator",
    73: "book",
    74: "clock",
    75: "vase",
    76: "scissors",
    77: "teddy bear",
    78: "hair drier",
    79: "toothbrush",
}

total_count = []
while True:
    success, img = cap.read()
    imgRegion = cv2.bitwise_and(img, mask)
    results = model(imgRegion, stream=True)
    detections = np.empty((0, 5))

    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            conf = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])
            if classNames[cls] in ['car', 'bus', 'truck', 'motorcycle', 'bicycle'] and conf > 0.3:
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 4)
                cvzone.putTextRect(img, f"{classNames[cls]}", (max(0, x1), max(35, y1)), scale=3, thickness=3, colorR=(255, 0, 0))

                currentArray = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, currentArray))

    result_tracker = tracker.update(detections)
    cv2.line(img, (line[0], line[1]), (line[2], line[3]), (0, 0, 255), 3)
    for result in result_tracker:
        x1, y1, x2, y2, id = result
        x1, y1, x2, y2, id = int(x1), int(y1), int(x2), int(y2), int(id)
        w, h = x2 - x1, y2 - y1
        cx, cy = x1+w // 2, y1 + h // 2
        cv2.circle(img, (cx, cy), 5, (255, 255, 255), cv2.FILLED)

        if line[0] < cx < line[2] and line[1]-15 < cy < line[1]+15 and total_count.count(id) == 0:
            total_count.append(id)
            cv2.line(img, (line[0], line[1]), (line[2], line[3]), (0, 255, 0), 3)
    cvzone.putTextRect(img, f'Total Count: {len(total_count)}', (50, 50), colorR=(0, 0, 255))

    imS = cv2.resize(img, (1280, 720))
    cv2.imshow("Image", imS)
    cv2.waitKey(1)
