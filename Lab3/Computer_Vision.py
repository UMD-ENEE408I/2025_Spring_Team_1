from ultralytics import YOLO
import cv2

cam = cv2.VideoCapture(0)  # Open webcam

model = YOLO("yolo-Weights/yolov8n.pt")

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
			  "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
			  "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
			  "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
			  "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
			  "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
			  "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
			  "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
			  "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
			  "teddy bear", "hair drier", "toothbrush"
			  ]

while cam.isOpened():
	ret, frame = cam.read()
	results = model(frame, stream=True)


	if not ret:
		break

	for object in results:
		boxes = object.boxes
		
		for box in boxes:
			x1,y1,x2,y2 = box.xyxy[0]
			if (classNames[int(box.cls[0])] == "apple" or classNames[int(box.cls[0])] == "sports ball"):
				#cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 255), 3)
				cv2.circle(frame, (int((x1+x2)/2), int((y1+y2)/2)), int((x2-x1)/2), (255, 0, 255), 3)

				#cv2.putText(frame, classNames[int(box.cls[0])], (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
				#cv2.putText(frame, str(round(box.conf[0].numpy() * 100, 2)) + "%", (int(x2)- 200, int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
			elif (classNames[int(box.cls[0])] == "person"):
				cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 255), 3)

		
	cv2.imshow("Yolo", frame)

	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

cam.release()
cv2.destroyAllWindows()