import torch
import numpy as np
import cv2
from modules.strong_sort.strong_sort import StrongSORT

tracker = StrongSORT(
    "data/osnet_x1_0_msmt17.pth",
    "cuda",
    max_dist=0.2,
    max_iou_distance=0.7,
    max_age=30,
    n_init=3,
    nn_budget=100,
    mc_lambda=0.995,
    ema_alpha=0.9,
)
        
model = torch.hub.load('yolov5','custom', path="data/detectionCalciatori.pt",force_reload=True,source='local')
cap = cv2.VideoCapture("gui/video_to_analyze/final_game.mp4")
ret,img = cap.read()
while ret:
    detection = model(img)
    detection_results = detection.pandas().xyxy[0]

    xywhs = []
    confidences =[]
    classes =[]
    for _, row in detection_results.iterrows():
        xywhs.append([int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])])
        confidences.append(row['confidence'])
        classes.append(row['class'])

    outputs = tracker.update(xywhs, confidences, classes, img)

    ids      = []
    classes  = []
    bboxes_o = []
    for output in outputs:
        ids.append(int(output[4]))
        classes.append(int(output[5]))
        bboxes_o.append(output[0:4])
    #bboxes_t = self._to_another_space(bboxes_o)
    print(bboxes_o)
    for bbox in bboxes_o:
        xmin, ymin, xmax, ymax = map(int, bbox)
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)  # Green rectangle, thickness=2

    cv2.imshow("img",img)
    cv2.waitKey(1)
    ret,img = cap.read()
