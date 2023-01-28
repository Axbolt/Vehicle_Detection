import cv2
import glob

# Static image vehicle detector

net = cv2.dnn.readNet("dnn_model/yolov4.weights", "dnn_model/yolov4.cfg")
model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(832, 832), scale=1 / 255)

# Allow only classes containing vehicles
c_allowed = [2, 3, 5, 7]

# Load images from a folder
images_folder = glob.glob("images/*.jpg")

# Loop through all the images
for img_path in images_folder:
    print("Img path", img_path)
    img = cv2.imread(img_path)

    # Detect vehicles
    vehicle_boxes = []
    class_ids, scores, boxes = model.detect(img, nmsThreshold=0.4)
    for class_id, score, box in zip(class_ids, scores, boxes):
        if score < 0.5:
            # Skip detection with low confidence
            continue

        if class_id in c_allowed:
            vehicle_boxes.append(box)

    for box in vehicle_boxes:
        x, y, w, h = box

        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 3)
        cv2.putText(img, f'{int(score * 100)}%', (x, y - 10),
                    cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

    img = cv2.resize(img, (800, 600), interpolation=cv2.INTER_AREA)
    cv2.imshow('Vehicle detection', img)
    if cv2.waitKey(3000) & 0xFF == ord('q'):
        break
