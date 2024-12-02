import torch

def load_model():
    # Load the pre-trained YOLOv5 model directly from the Ultralytics GitHub repository
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', source='github')  # 'yolov5s' is a small, fast model variant
    return model

def detect_objects(model, frame):
    # Perform object detection on the frame
    results = model(frame)
    return results
