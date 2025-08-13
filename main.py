from ultralytics import YOLO
import cv2
import yaml
import time

model = YOLO("best.pt")
print("Model loaded")


with open("data.yaml", "r") as f:
    data_yaml = yaml.safe_load(f)
class_names = data_yaml['names']
print("Classes loaded:", class_names)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot access webcam")
    exit()


cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

detected_text = ""
last_detected = ""
last_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break


    frame_resized = cv2.resize(frame, (640, 480))


    results = model.predict(source=frame_resized, conf=0.25, verbose=False)

    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            letter = class_names[cls_id]


            if letter != last_detected and time.time() - last_time > 1:
                detected_text += letter
                last_detected = letter
                last_time = time.time()


            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame_resized, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame_resized, letter, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)  

    
    cv2.putText(frame_resized, f"Text: {detected_text}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    cv2.imshow("Sign Detection (Mini Window)", frame_resized)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):
        detected_text = ""

cap.release()
cv2.destroyAllWindows()
