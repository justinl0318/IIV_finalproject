from ultralytics import YOLO
import cv2

# Load the YOLO model
model = YOLO('yolov8n.pt')  

results = model("./car.jpg")

# # Open the video file
# cap = cv2.VideoCapture(VIDEO_PATH)

# # Loop through the frames
# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break
    
#     # Perform YOLO detection on the frame
#     results = model(frame)
    
#     # Visualize the results on the frame
#     annotated_frame = results[0].plot()

#     # Display the annotated frame
#     cv2.imshow("YOLOv8 Inference", annotated_frame)
    


# # Release the video capture object and close all windows
# cap.release()
# cv2.destroyAllWindows()
