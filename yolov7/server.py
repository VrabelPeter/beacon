from flask import Flask, request, jsonify
import cv2
import torch
from torchvision import transforms, models
from PIL import Image
import numpy as np
import io
import time
import threading
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from msrest.authentication import CognitiveServicesCredentials

from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_coords
from utils.datasets import letterbox
from utils.plots import plot_one_box

# Initialize Flask app
app = Flask(__name__)


def load_yolov7_model(weights_path='yolov7.pt'):
    model = attempt_load(weights_path, map_location=torch.device('cpu'))
    return model


def preprocess_yolo_image(image, img_size=640):
    img = letterbox(image, img_size)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).float()
    img /= 255.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    return img


def run_yolov7_on_image(frame, model, img_size=640, conf_thres=0.25, iou_thres=0.45):
    img = preprocess_yolo_image(frame, img_size)
    with torch.no_grad():
        pred = model(img, augment=False)[0]
    pred = non_max_suppression(pred, conf_thres, iou_thres)
    for det in pred:
        if len(det):
            det[:, :4] = scale_coords(
                img.shape[2:], det[:, :4], frame.shape).round()
    return pred


def suggest_path(yolo_results, frame_width):
    center_threshold = frame_width * 0.4
    obstacle_left = False
    obstacle_right = False
    front_clear = True

    for det in yolo_results:
        if det is not None and len(det):
            for *xyxy, conf, cls in det:
                xmin, ymin, xmax, ymax = xyxy
                object_center = (xmin + xmax) / 2

                # Check if object is in front (low y-value)
                if ymin < frame_width * 0.2:
                    front_clear = False

                if object_center < center_threshold:  # Left side detection
                    obstacle_left = True
                # Right side detection
                elif object_center > (frame_width - center_threshold):
                    obstacle_right = True

    if front_clear and obstacle_left and obstacle_right:
        return "Space clear, stay center"
    elif obstacle_left and not obstacle_right:
        return "Move right"
    elif obstacle_right and not obstacle_left:
        return "Move left"
    elif obstacle_left and obstacle_right:
        return "Obstacle ahead"
    else:
        return None


def setup_places365():
    model_places = models.resnet50(pretrained=False)
    model_places.fc = torch.nn.Linear(2048, 365)
    checkpoint = torch.load('resnet50_places365.pth.tar',
                            map_location=torch.device('cpu'))
    new_state_dict = {}
    for k, v in checkpoint['state_dict'].items():
        new_state_dict[k[7:]] = v if k.startswith('module.') else v
    model_places.load_state_dict(new_state_dict)
    model_places.eval()
    categories = []
    with open('categories_places365.txt') as class_file:
        categories = [line.strip().split(' ')[0][3:] for line in class_file]
    return model_places, categories


def run_places_on_image(model_places, categories, frame):
    transform_places = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225]),
    ])
    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    img_transformed = transform_places(img_pil).unsqueeze(0)

    with torch.no_grad():
        output = model_places(img_transformed)

    probs = torch.nn.functional.softmax(output[0], dim=0)
    top5_prob, top5_catid = torch.topk(probs, 5)
    scene = categories[top5_catid[0]]
    return scene


# Initialize models and other global variables
# Load models
yolo_model = load_yolov7_model(weights_path='yolov7.pt')
model_places, categories = setup_places365()
class_names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
               'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
               'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
               'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
               'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
               'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tvmonitor', 'laptop',
               'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
               'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
               'toothbrush']


# Azure credentials (replace with your own)
subscription_key = ''
endpoint = ''

# Initialize Azure client
computervision_client = ComputerVisionClient(
    endpoint, CognitiveServicesCredentials(subscription_key))


def collect_detected_objects(det, class_names):
    detected_objects = set()
    for *xyxy, conf, cls in det:
        label = class_names[int(cls)]
        detected_objects.add(label)
    return list(detected_objects)


def recognize_text_azure_frame(frame, computervision_client: ComputerVisionClient):
    # Encode the frame as JPEG
    ret, encoded_image = cv2.imencode('.jpg', frame)
    image_bytes = encoded_image.tobytes()

    # Send image_bytes to Azure OCR
    read_response = computervision_client.read_in_stream(
        io.BytesIO(image_bytes), raw=True)

    # Get the operation location (URL with an ID at the end)
    operation_location = read_response.headers["Operation-Location"]
    # Extract the operation ID
    operation_id = operation_location.split("/")[-1]

    # Wait for the operation to complete
    while True:
        read_result = computervision_client.get_read_result(operation_id)
        if read_result.status.lower() not in ['notstarted', 'running']:
            break
        time.sleep(1)

    # Parse and return the results
    text_results = ""
    if read_result.status.lower() == 'succeeded':
        for page in read_result.analyze_result.read_results:
            for line in page.lines:
                text_results += line.text + " "
    else:
        print("OCR operation failed.")

    return text_results.strip()


@app.route('/navigation', methods=['POST'])
def navigation():
    try:
        # Receive image from client
        image_file = request.files['image']
        image_bytes = image_file.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Process the image frame
        yolo_results = run_yolov7_on_image(frame, yolo_model)
        path_suggestion = suggest_path(yolo_results, frame.shape[1])

        # Perform scene recognition
        scene_label = run_places_on_image(model_places, categories, frame)

        # Prepare response
        response = {
            'path_suggestion': path_suggestion,
            'scene_label': scene_label,
            'detected_objects': []
        }

        # Extract detected objects
        for det in yolo_results:
            if det is not None and len(det):
                for *xyxy, conf, cls in det:
                    label = class_names[int(cls)]
                    response['detected_objects'].append(label)
                    # You can include more details if needed

        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/text_recognition', methods=['POST'])
def text_recognition():
    try:
        # Receive image from client
        image_file = request.files['image']
        image_bytes = image_file.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Save image temporarily if needed
        # image_path = 'temp_image.jpg'
        # cv2.imwrite(image_path, frame)

        # Perform text recognition using Azure OCR
        recognized_text = recognize_text_azure_frame(
            frame, computervision_client)

        # Prepare response
        response = {
            'recognized_text': recognized_text
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    # Run the Flask app on all network interfaces
    app.run(host='0.0.0.0', port=5000)
