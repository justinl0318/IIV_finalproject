from ultralytics import YOLO
import cv2

# Load the YOLO model
model = YOLO('yolov8n.pt')  

# Perform detection
results = model("./canvas.png")

# Retrieve the annotated image from the results
annotated_image = results[0].plot() 

# Save the annotated image
output_path = "./output.png"
cv2.imwrite(output_path, annotated_image)

print(f"Results saved to {output_path}")
