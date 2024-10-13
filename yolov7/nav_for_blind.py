import cv2
import torch
from torchvision import transforms, models
from PIL import Image
import numpy as np
import os
import platform
import pyttsx3
import time
import threading
import speech_recognition as sr
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from msrest.authentication import CognitiveServicesCredentials

# Add YOLOv7 and utility paths if needed
# sys.path.append('/path/to/yolov7')  # Update this path if necessary

from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_coords
from utils.datasets import letterbox
from utils.plots import plot_one_box

# Initialize text-to-speech engine
engine = pyttsx3.init()

# Initialize global variables
announced_objects = set()
announced_scene = None  # Keep track of the last announced scene
wake_word_detected = threading.Event()

# Azure credentials (replace with your own)
subscription_key = ''
endpoint = ''

# Initialize Azure client
computervision_client = ComputerVisionClient(
    endpoint, CognitiveServicesCredentials(subscription_key))


def play_alert_sound():
    if platform.system() == 'Windows':
        import winsound
        winsound.Beep(1000, 500)
    else:
        os.system('say "Alert"')


def announce_object(label):
    if label not in announced_objects:
        announced_objects.add(label)
        engine.say(f'{label} detected')
        engine.runAndWait()


def announce_scene(scene):
    global announced_scene
    if scene != announced_scene:
        announced_scene = scene
        engine.say(f'The scene is {scene}')
        engine.runAndWait()


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


def check_proximity(xyxy, frame_width, frame_height, threshold=0.3):
    xmin, ymin, xmax, ymax = xyxy
    box_width = xmax - xmin
    box_height = ymax - ymin
    if box_width > threshold * frame_width or box_height > threshold * frame_height:
        return True
    return False


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


def plot_boxes_on_image(frame, yolo_results, class_names):
    frame_height, frame_width, _ = frame.shape
    for det in yolo_results:
        if det is not None and len(det):
            for *xyxy, conf, cls in det:
                label = f'{class_names[int(cls)]} {conf:.2f}'
                if check_proximity(xyxy, frame_width, frame_height):
                    play_alert_sound()
                announce_object(class_names[int(cls)])
                plot_one_box(xyxy, frame, label=label,
                             color=(0, 255, 0), line_thickness=2)
    return frame


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


def listen_for_wake_word():
    recognizer = sr.Recognizer()
    microphone = sr.Microphone()
    while True:
        with microphone as source:
            print("Listening for wake word...")
            recognizer.adjust_for_ambient_noise(source)
            try:
                audio = recognizer.listen(source, timeout=5)
                speech_text = recognizer.recognize_google(audio).lower()
                print(f"You said: {speech_text}")
                if "wake up" in speech_text:
                    print("Wake word detected!")
                    wake_word_detected.set()
            except sr.WaitTimeoutError:
                print("Timeout, no speech detected.")
            except sr.UnknownValueError:
                print("Could not understand audio.")
            except sr.RequestError as e:
                print(f"Speech Recognition error; {e}")


def recognize_text_azure(image_path, computervision_client: ComputerVisionClient):
    with open(image_path, "rb") as image_stream:
        read_response = computervision_client.read_in_stream(
            image_stream, raw=True)

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

# Real-time object detection and scene recognition with wake word listener


def real_time_detection_with_wake_word():
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

    cap = cv2.VideoCapture(0)
    frame_count = 0

    # Start wake word listener in a separate thread
    wake_word_listener = threading.Thread(target=listen_for_wake_word)
    wake_word_listener.daemon = True
    wake_word_listener.start()

    while cap.isOpened():
        start_time = time.time()

        # Check if wake word was detected
        if wake_word_detected.is_set():
            # Pause navigation mode
            print("Pausing navigation mode for text recognition.")
            engine.say("Please hold still for text recognition.")
            engine.runAndWait()

            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame for text recognition.")
                continue
            image_path = "captured_image.jpg"
            cv2.imwrite(image_path, frame)

            # Perform text recognition using Azure OCR
            recognized_text = recognize_text_azure(
                image_path, computervision_client)
            if recognized_text:
                print(f"Recognized Text: {recognized_text}")
                engine.say(recognized_text)
                engine.runAndWait()
            else:
                print("No text recognized.")
                engine.say("No text recognized.")
                engine.runAndWait()

            # Reset wake word detected flag
            wake_word_detected.clear()
            print("Resuming navigation mode.")
            # Reset announced objects to avoid repeat alerts
            announced_objects.clear()
            continue

        ret, frame = cap.read()
        if not ret:
            break

        # Object Detection with YOLOv7
        yolo_results = run_yolov7_on_image(frame, yolo_model)
        path_suggestion = suggest_path(yolo_results, frame.shape[1])

        if frame_count % 5 == 0:
            # Scene Detection with Places365
            scene_label = run_places_on_image(model_places, categories, frame)
            announce_scene(scene_label)

        # Plot YOLOv7 results
        frame = plot_boxes_on_image(frame, yolo_results, class_names)

        if frame_count % 5 == 0:
            cv2.putText(frame, f'Scene: {scene_label}', (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # Announce path suggestion if needed
        if path_suggestion:
            engine.say(path_suggestion)
            engine.runAndWait()

        # Display frame rate (FPS)
        end_time = time.time()
        fps = 1 / (end_time - start_time)
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Show the frame
        cv2.imshow('Navigation Mode', frame)

        # Exit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    real_time_detection_with_wake_word()
