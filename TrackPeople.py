from os import path
from Visitor import Visitor
import cv2
import time
from ultralytics import YOLO
import numpy as np
from deepface import DeepFace

model = YOLO("yolo11n.pt")

cap = cv2.VideoCapture(0)

visitors = []
visitors_index = 0

def get_center(x1, y1, x2, y2):
    cx = (x1 + x2) / 2
    cy = (y1 +y2) / 2
    return int(cx), int(cy)

frame_num = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.track(frame, persist=True, classes=[0], verbose=False)
    detections = results[0].boxes

    for box in results[0].boxes:
        if box.id is not None:
            person_id = int(box.id)
            curr_visitor = None;
        
            # Check if this person_id already exists in visitors list
            existing_visitor = None
            for visitor in visitors:
                if visitor.id == person_id:  # Assuming your Visitor class has an 'id' attribute
                    existing_visitor = visitor
                    curr_visitor = visitor
                    break
        
            # Get bounding box coordinates first (needed for cropping)
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # If not found, create new visitor and add to list
            if existing_visitor is None:
                new_visitor = Visitor(id=person_id, frame_shape=frame.shape)  # Add whatever parameters your Visitor needs
                visitors.append(new_visitor)
                print(f"New visitor added: ID {person_id}")
                curr_visitor = new_visitor

            # Only analyze new visitors
            if not curr_visitor.analyzed:
                try:
                    # Crop the person from the frame
                    person_crop = frame[y1:y2, x1:x2]
                    if person_crop.size > 0:  # Make sure crop is valid
                        result = DeepFace.analyze(person_crop, actions=['age', 'gender'], enforce_detection=False)
                        curr_visitor.age = result[0]['age']
                        curr_visitor.gender = result[0]['dominant_gender']
                        curr_visitor.analyzed = True
                except Exception as e:
                    print(f"Could not analyze visitor {curr_visitor.id}: {e}")
        
            # Draw the bounding box and circle
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cx, cy = get_center(x1, y1, x2, y2)
            cv2.circle(frame, (cx, cy), 1, (0, 0, 255), 10)

            # Store center point data every second
            if frame_num % 60 == 0:
                curr_visitor.path.append((cx, cy))
                curr_visitor.add_point(cx, cy)

        
            # Optional: Add the ID as a label
            cv2.putText(frame, f"ID: {curr_visitor.id}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
    frame_num += 1

        
    cv2.imshow("Exhibit Tracking", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()
for vis in visitors:
    id = vis.id
    path = vis.path
    age = vis.age
    gender = vis.gender
    if (len(vis.path) < 2):
        del vis
        break
    print(id)
    print(path)
    print(age)
    print(gender)
    filename = str(vis.id) + "_path.png"
    cv2.imwrite(filename, vis.get_path_image())

