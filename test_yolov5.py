import torch
from PIL import Image

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Load your local image
img_path = 'cat_dog.jpg'  # Replace with your image file name
img = Image.open(img_path)

# Perform object detection
results = model(img)

# Print results
results.print()

# Display results
results.show()

# Save results
results.save('results')  # Save results in a folder named 'results'
