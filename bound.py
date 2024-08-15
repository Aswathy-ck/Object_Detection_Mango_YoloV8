from ultralytics import YOLO
import cv2

# Define model and image paths
model_path = 'C:\\Users\\ckasw\\OneDrive\\Desktop\\PROJECT INTERNSHIP\\YOLOV8\\runs\\detect\\train4\\weights\\last.pt'
image_path = 'C:\\Users\\ckasw\\Downloads\\alphonso.jpg'


# Load image
img = cv2.imread(image_path)

# Load the YOLO model
model = YOLO(model_path)

# Run inference on the image
results = model(image_path)

# Iterate over the results and draw bounding boxes
for result in results:
    for box in result.boxes.xyxy:  # Extract bounding box coordinates
        x1, y1, x2, y2 = map(int, box[:4])  # Convert to int
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Draw rectangle with blue color

    if hasattr(result, 'keypoints') and result.keypoints is not None:  # Ensure that keypoints attribute exists and is not None
        for keypoint_index, keypoint in enumerate(result.keypoints.tolist()):
            cv2.putText(img, str(keypoint_index), (int(keypoint[0]), int(keypoint[1])),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

# Display the image with bounding boxes and keypoints
cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
