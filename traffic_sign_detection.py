import numpy as np
from roboflow import Roboflow
import supervision as sv
import cv2

# Initialize Roboflow
rf = Roboflow(api_key="BxgGN2VV9dHmazRhr3eu")
project = rf.workspace().project("roboflow-tenyks-fix-your-dataset")
model = project.version(2).model

# Perform the prediction
result = model.predict("/Users/abhinav/Downloads/stop_sign.jpg", confidence=40, overlap=30).json()

# Extract labels, bounding boxes, and class_ids
labels = [item["class"] for item in result["predictions"]]
boxes = np.array([[
    item['x'] - item['width'] / 2,  # x1
    item['y'] - item['height'] / 2,  # y1
    item['x'] + item['width'] / 2,  # x2
    item['y'] + item['height'] / 2   # y2
] for item in result["predictions"]])
class_ids = np.array([item["class_id"] for item in result["predictions"]])  # Add class_ids

# Convert to Supervision's Detections object, including both bounding boxes and class_ids
detections = sv.Detections(xyxy=boxes, class_id=class_ids)

# Annotators
label_annotator = sv.LabelAnnotator()
bounding_box_annotator = sv.BoxAnnotator()

# Load the image
image = cv2.imread("/Users/abhinav/Downloads/stop_sign.jpg")

# Annotate the image with bounding boxes and labels
annotated_image = bounding_box_annotator.annotate(scene=image, detections=detections)
annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)

# Display the annotated image
sv.plot_image(image=annotated_image, size=(16, 16))
