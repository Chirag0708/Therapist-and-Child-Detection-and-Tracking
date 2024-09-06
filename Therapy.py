import cv2
import torch
from yolov5 import detect
from deep_sort_realtime.deepsort_tracker import DeepSort

# Initialize YOLOv5 model (pre-trained on COCO dataset)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Initialize DeepSORT for tracking
tracker = DeepSort(max_age=30, n_init=3, nn_budget=100)

# Load the test video
video_path = 'path_to_test_video.mp4'
cap = cv2.VideoCapture(video_path)

# Video writer for saving output
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output_video.mp4', fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Perform detection
    results = model(frame)
    detections = results.xyxy[0].cpu().numpy()  # (x1, y1, x2, y2, confidence, class)

    # Filter detections: Only keep person class (class 0)
    person_detections = [det for det in detections if int(det[5]) == 0]

    # Prepare detections for tracker
    boxes = [[det[0], det[1], det[2], det[3]] for det in person_detections]
    scores = [det[4] for det in person_detections]

    # Track the detections
    tracks = tracker.update_tracks(boxes, scores, frame=frame)

    # Draw bounding boxes and IDs
    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        ltrb = track.to_ltrb()
        x1, y1, x2, y2 = map(int, ltrb)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f'ID: {track_id}', (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    # Write the frame to the output video
    out.write(frame)

    # Optional: Display the frame (comment out if running on a server)
    # cv2.imshow('Frame', frame)
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #    break

cap.release()
out.release()
cv2.destroyAllWindows()
