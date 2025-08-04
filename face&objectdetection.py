import cv2
import time

# Load classifiers
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_smile.xml")

# Start webcam
cap = cv2.VideoCapture(0)

# Flag to save smile only once
smile_captured = False

while True:
    ret, frame = cap.read()
    gray_scale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detect_face = face_cascade.detectMultiScale(gray_scale, 1.1, 5)

    if len(detect_face) == 0:
        cv2.putText(frame, "No Face Detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    for (x, y, w, h) in detect_face:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        roi_gray = gray_scale[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]

        eye = eye_cascade.detectMultiScale(roi_gray, 1.1, 10)
        if len(eye) > 0:
            cv2.putText(frame, "Eye Detected", (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

        smile = smile_cascade.detectMultiScale(roi_gray, 1.8, 20)
        if len(smile) > 0:
            cv2.putText(frame, "Smiling...", (x, y - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            # Save only once
            if not smile_captured:
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                cv2.imwrite(f"image_capture_{timestamp}.jpg", frame)
                print(f"image captured at {timestamp}")
                smile_captured = True  # Prevent further saving

    # Display result
    cv2.imshow("Face Detection", frame)

    # Press 'r' to reset capture flag and allow another save
    if cv2.waitKey(1) & 0xFF == ord('r'):
        smile_captured = False
        print("Ready for next smile capture...")

    # Press 'w' to exit
    if cv2.waitKey(1) & 0xFF == ord('w'):
        break

cap.release()
cv2.destroyAllWindows()        
