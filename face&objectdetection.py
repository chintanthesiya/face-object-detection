import cv2

face_cascade=cv2.CascadeClassifier("firstproject\opecvlib\haarcascade_frontalface_default.xml")
eye_cascade=cv2.CascadeClassifier("firstproject\opecvlib\haarcascade_eye.xml")
smile_cascade=cv2.CascadeClassifier("firstproject\opecvlib\haarcascade_smile.xml")

cap=cv2.VideoCapture(0)

while True:
    ret,frame=cap.read()
    gray_scale=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    detect_face=face_cascade.detectMultiScale(gray_scale,1.1,5)

    for (x,y,w,h) in detect_face:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

    roi_gray=gray_scale[y:y+h,x:x+w]  #region of intrest for face detection
    roi_color=frame[y:y+h,x:x+w]    

    eye=eye_cascade.detectMultiScale(roi_gray,1.1,10)
    if len(eye) > 0:
        cv2.putText(frame, "Eye Detected", (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    smile=smile_cascade.detectMultiScale(roi_gray,1.8,20)
    if len(smile) > 0:  
        cv2.putText(frame, "Smiling...", (x, y - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0, 255), 2)    

    cv2.imshow("Face Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('w'):
        break
cap.release()
cv2.destroyAllWindows()         
