from ultralytics import YOLO

# Load a model
model = YOLO('yolov8n.pt')

# Train the model
results = model.train(data='yolo_train_data.yaml', epochs=60, device='cpu')
