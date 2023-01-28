import cv2
import time

# Video vehicle detector

net = cv2.dnn.readNet("dnn_model/yolov4.weights", "dnn_model/yolov4.cfg")
model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(288, 288), scale=1 / 255)

# Allow only classes containing vehicles
c_allowed = [2, 3, 5, 6, 7]

cap = cv2.VideoCapture('Car - 2165.mp4')
# cap = cv2.VideoCapture('Street - 19627.mp4')

while True:

    isTrue, frame = cap.read()
    start = time.time()

    if not isTrue:
        print("Ignoring empty camera frame.")
        break

    # Detect vehicles
    vehicle_boxes = []
    class_ids, scores, boxes = model.detect(frame, nmsThreshold=0.4)
    for class_id, score, box in zip(class_ids, scores, boxes):
        if score < 0.5:
            # Skip detection with low confidence
            continue

        if class_id in c_allowed:
            vehicle_boxes.append(box)

    for box in vehicle_boxes:
        x, y, w, h = box

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 3)
        cv2.putText(frame, f'{int(score * 100)}%', (x, y - 10),
                    cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

    end = time.time()
    totalTime = end - start

    fps = 1 / totalTime

    frame = cv2.resize(frame, (960, 540), interpolation=cv2.INTER_AREA)
    cv2.putText(frame, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
    cv2.imshow('Vehicle detection', frame)
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()