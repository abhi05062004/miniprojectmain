from ultralytics import YOLO

# Load the trained YOLO model
model_path = "vegetable.pt"  # Change this to your actual model path
model = YOLO(model_path)

# Provide the image path
image_path = r"D:\project\hh.jpg"  # Change this to the image you want to detect

# Perform detection
results = model(image_path)  

# Loop through results and print detected objects
for result in results:
    for box in result.boxes:
        cls = int(box.cls[0])  # Class ID
        conf = float(box.conf[0])  # Confidence score
        print(f"Detected: {model.names[cls]} with confidence: {conf:.2f}")

# Show the image with detected objects
result.show()