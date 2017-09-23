from goprocam import GoProCamera
from goprocam import constants
import cv2
import numpy as np
gpCam = GoProCamera.GoPro()
#gpCam.stream("udp://localhost:8081")
#gopro.livestream("start")
#gopro.stream("udp://localhost:5000")
print(gpCam.getMedia())
gpCam.stream("udp://127.0.0.1:10000")

cascPath="haarcascades_AlexeyAB\haarcascades_GPU\haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)
#gpCam = GoProCamera.GoPro()
cap = cv2.VideoCapture("udp://127.0.0.1:10000")
font = cv2.FONT_HERSHEY_PLAIN
video = cv2.VideoWriter("tester.avi", cv2.CV_FOURCC('M','J','P','G'), 10.0,(w, h),True)

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, "This one!", (x+w, y+h), font, 0.8, (0, 255, 0), 1, cv2.LINE_AA)
        video.write(frame)
    cv2.imshow("GoPro OpenCV", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
video.release()
cap.release()
cv2.destroyAllWindows()