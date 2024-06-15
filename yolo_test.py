import cv2 as cv
import cvzone
from ultralytics import YOLO

capture = cv.VideoCapture(r'C:\Users\sarth\OneDrive\Documents\Internship Assignmet\AIMSDTU\demo.mp4')

model_path = r'C:\Users\sarth\OneDrive\Documents\Internship Assignmet\AIMSDTU\yolov8n-face.pt'
facemodel = YOLO(model_path)

while True:
    isTrue, frame = capture.read()
    frame = cv.resize(frame,(1020,720))
    result = facemodel.predict(frame,conf = 0.45)
    for info in result:
        parameters = info.boxes
        for box in parameters:
            x1,y1,x2,y2 = box.xyxy[0]
            x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)
            h,w = y2 - y1, x2 - x1
            cvzone.cornerRect(frame,[x1,y1,w,h],l=9,rt=3)
    
    cv.imshow('Final', frame)

    if cv.waitKey(20) & 0xFF == ord('d'):
        break

capture.release()
cv.destroyAllWindows()