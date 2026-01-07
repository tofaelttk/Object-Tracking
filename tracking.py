import cv2
import numpy as np
import sqlite3
from datetime import datetime
import math


conn = sqlite3.connect('object_tracking.db')
c = conn.cursor()
c.execute('''
    CREATE TABLE IF NOT EXISTS tracking_data (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        object_id INTEGER,
        object_name TEXT,
        timestamp TEXT,
        centroid_x REAL,
        centroid_y REAL,
        speed REAL,
        acceleration REAL,
        angle REAL,
        total_distance REAL
    )
''')
conn.commit()



yolo_net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]


cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise Exception("Could not open video device")

fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=False)

tracked_objects = {}  # key: object_id, value: dictionary with tracking info
next_object_id = 0


font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.5
font_color = (0, 255, 0)
thickness = 1


def calculate_metrics(obj, current_centroid, current_time):
    """
    Given the previous object info and the current centroid,
    compute the speed (pixels/sec), acceleration (pixels/sec²),
    angle (degrees) and the distance traveled.
    """
    time_diff = (current_time - obj['last_time']).total_seconds()
    if time_diff == 0:
        return 0, 0, 0, 0
    dx = current_centroid[0] - obj['last_centroid'][0]
    dy = current_centroid[1] - obj['last_centroid'][1]
    distance = math.sqrt(dx**2 + dy**2)
    speed = distance / time_diff
    acceleration = (speed - obj['last_speed']) / time_diff
    # Calculate angle (adjust for image coordinate system: y increases downward)
    angle = math.degrees(math.atan2(-dy, dx))
    angle = (angle + 360) % 360  # normalize to 0-360°
    return speed, acceleration, angle, distance

while True:
    ret, frame = cap.read()
    if not ret:
        break

    current_time = datetime.now()
    

    height, width = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    yolo_net.setInput(blob)
    layer_names = yolo_net.getLayerNames()
    # Get the output layer names in a format that works regardless of OpenCV version
    try:
        output_layers = [layer_names[i - 1] for i in yolo_net.getUnconnectedOutLayers().flatten()]
    except AttributeError:
        # Fallback for older OpenCV versions
        output_layers = [layer_names[i[0] - 1] for i in yolo_net.getUnconnectedOutLayers()]
    
    outs = yolo_net.forward(output_layers)
    
    yolo_detections = []  # List of tuples: (label, (x, y, w, h))
    conf_threshold = 0.5
    nms_threshold = 0.4
    boxes = []
    confidences = []
    class_ids = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > conf_threshold:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
    if len(indices) > 0:
        for i in indices.flatten():
            box = boxes[i]
            x, y, w, h = box
            label = classes[class_ids[i]]
            yolo_detections.append((label, (x, y, w, h)))
            # Draw YOLO detection boxes in magenta
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 255), 2)
            cv2.putText(frame, label, (x, y-10), font, 0.5, (255, 0, 255), 2)
  
    fgmask = fgbg.apply(frame)
    thresh = cv2.threshold(fgmask, 127, 255, cv2.THRESH_BINARY)[1]
    kernel = np.ones((5, 5), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    
    # Find contours of moving objects
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    current_centroids = []  # list of tuples: (centroid, bounding_box)
    
    for contour in contours:
        if cv2.contourArea(contour) < 500:
            continue  # skip small areas (noise)
        x, y, w, h = cv2.boundingRect(contour)
        centroid = (int(x + w/2), int(y + h/2))
        current_centroids.append((centroid, (x, y, w, h)))
        # Draw bounding box and centroid (from motion detection) in green
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.circle(frame, centroid, 4, (0, 0, 255), -1)
    

    updated_ids = []
    for item in current_centroids:
        centroid, bbox = item
        min_dist = float('inf')
        obj_id = None
        # Try to match the detected centroid with an existing object
        for oid, obj in tracked_objects.items():
            dist = math.hypot(centroid[0] - obj['last_centroid'][0],
                              centroid[1] - obj['last_centroid'][1])
            if dist < min_dist and dist < 50:  # threshold for matching
                min_dist = dist
                obj_id = oid
                
        if obj_id is not None:
            # Existing object: compute metrics and update its info
            speed, accel, angle, distance = calculate_metrics(tracked_objects[obj_id], centroid, current_time)
            new_total_distance = tracked_objects[obj_id]['total_distance'] + distance
            object_name = tracked_objects[obj_id]['object_name']
            tracked_objects[obj_id].update({
                'last_centroid': centroid,
                'last_speed': speed,
                'last_time': current_time,
                'total_distance': new_total_distance,
                'angle': angle,
                'acceleration': accel
            })
            updated_ids.append(obj_id)
            # Insert updated info into the database
            c.execute('''
                INSERT INTO tracking_data (object_id, object_name, timestamp, centroid_x, centroid_y, speed, acceleration, angle, total_distance)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (obj_id, object_name, current_time.isoformat(), centroid[0], centroid[1],
                  speed, accel, angle, new_total_distance))
        else:
            # New object: create entry and assign a default name (will be updated by YOLO if possible)
            this_id = next_object_id
            object_name = f"Object_{this_id}"
            tracked_objects[this_id] = {
                'object_name': object_name,
                'last_centroid': centroid,
                'last_speed': 0,
                'last_time': current_time,
                'total_distance': 0,
                'angle': 0,
                'acceleration': 0
            }
            updated_ids.append(this_id)
            c.execute('''
                INSERT INTO tracking_data (object_id, object_name, timestamp, centroid_x, centroid_y, speed, acceleration, angle, total_distance)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (this_id, object_name, current_time.isoformat(), centroid[0], centroid[1], 0, 0, 0, 0))
            next_object_id += 1
            
    # Remove objects that were not updated this frame (lost objects)
    lost_ids = [oid for oid in list(tracked_objects.keys()) if oid not in updated_ids]
    for oid in lost_ids:
        del tracked_objects[oid]
        

    for oid, obj in tracked_objects.items():
        centroid = obj['last_centroid']
        for detection in yolo_detections:
            label, (dx, dy, dw, dh) = detection
            if dx <= centroid[0] <= dx+dw and dy <= centroid[1] <= dy+dh:
                tracked_objects[oid]['object_name'] = label
                break  # assign the first matching label
    

    current_time_str = current_time.strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(frame, f"Current Time: {current_time_str}", (10, 20), font, 0.6, (0, 255, 255), 2)
    
    y_offset = 40
    for oid, obj in tracked_objects.items():
        centroid = obj['last_centroid']
        # Draw an angle line to indicate movement direction
        line_length = 50
        angle_rad = math.radians(obj['angle'])
        end_point = (int(centroid[0] + line_length * math.cos(angle_rad)),
                     int(centroid[1] - line_length * math.sin(angle_rad)))
        cv2.line(frame, centroid, end_point, (255, 0, 0), 2)
        
        info = (f"ID: {oid} | {obj['object_name']} | Centroid: {centroid} | "
                f"Speed: {obj['last_speed']:.1f}px/s | Accel: {obj['acceleration']:.1f}px/s² | "
                f"Angle: {obj['angle']:.1f}° | Dist: {obj['total_distance']:.1f}px | "
                f"Time: {obj['last_time'].strftime('%H:%M:%S')}")
        cv2.putText(frame, info, (10, y_offset), font, font_scale, font_color, thickness, cv2.LINE_AA)
        y_offset += 20


    cv2.imshow('Object Tracking', frame)
    # (Optional) Show the threshold mask for debugging
    cv2.imshow('Threshold', thresh)
    
    # Exit when ESC key is pressed
    if cv2.waitKey(1) == 27:
        break

conn.commit()
cap.release()
cv2.destroyAllWindows()
conn.close()
