from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
import cv2 as cv
from collections import defaultdict
import numpy as np

# Load YOLO model
model = YOLO("yolov8n.pt")
print(model.names)

# Store the track history
track_history = defaultdict(lambda: [])

def draw_boxes(frame, boxes):
    """Draw detected bounding boxes on image frame"""

    # Create annotator object
    annotator = Annotator(frame)
    for box in boxes:
        class_id = box.cls
        class_name = model.names[int(class_id)]
        coordinator = box.xyxy[0]
        confidence = box.conf
    # Draw bounding box
        annotator.box_label(
        box=coordinator, label=class_name, color=colors(18, False)
    )
    return annotator.result()


def detect_object(frame):
    """Detect object from image frame"""

    # Detect object from image frame
    results = model.predict(frame, classes=[15])
    

    for result in results:
        frame = draw_boxes(frame, result.boxes)
    
    return frame


if __name__ == "__main__":
    video_path = "CatZoomies.mp4"
    cap = cv.VideoCapture(video_path)


    while cap.isOpened():
        # Read image frame
        ret, frame = cap.read()
    

        ## Line tracking Use Plotting Tracks Over Time From Ultralytics Docs
        result = model.track(frame, persist=True)[0]
        if result.boxes and result.boxes.is_track:
            boxes = result.boxes.xywh.cpu()
            track_ids = result.boxes.id.int().cpu().tolist()
            # Plot the tracks
            for box, track_id in zip(boxes, track_ids):
                x, y, w, h = box
                track = track_history[track_id]
                track.append((float(x), float(y)))  # x, y center point
                if len(track) > 30:  # retain 30 tracks for 30 frames
                    track.pop(0)
                # Draw the tracking lines
                points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                cv.polylines(frame, [points], isClosed=False, color=(230, 230, 230), thickness=5)


        if ret:
            # Detect motorcycle from image frame
            frame_result = detect_object(frame)

            # Show result
            cv.putText(frame_result,"Apivit-Clicknext-Internship-2025",(700,50),cv.FONT_HERSHEY_SIMPLEX,1,(	(0, 0, 255)),2)
            cv.namedWindow("Video", cv.WINDOW_NORMAL)
            cv.imshow("Video", frame_result)
            cv.waitKey(30)

        else:
            break
            
    cap.release()
    cv.destroyAllWindows()




