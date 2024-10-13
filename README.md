# Beacon - AI Assistant for the Visually Impaired

## Project Overview

**Beacon** is an AI-powered assistant designed to help visually impaired individuals navigate their surroundings with confidence. Beacon integrates real-time object detection, scene recognition, and OCR (Optical Character Recognition) using state-of-the-art models such as YOLOv7 and Places365, along with an intuitive voice interface to provide users with actionable information about their environment. It is built to foster independence by informing users of nearby objects, hazards, and important textual details such as signs, making daily activities more accessible.

---

## Innovation, Creativity, and Impact

**Creativity**:  
Beacon is a unique blend of real-time object detection, scene recognition, and OCR, built specifically for visually impaired individuals. The system is designed to provide voice feedback based on the context of the user's surroundings. While object detection and OCR technology are not new, combining these capabilities in a mobile-friendly application with a user-centric interface offers a creative, practical solution to accessibility issues. It goes beyond conventional solutions by integrating multiple AI-powered technologies to provide contextually aware feedback.

**Impact**:  
If implemented in the real world, Beacon could significantly improve the daily experiences of visually impaired individuals by enhancing their ability to navigate and understand their surroundings independently. This could be applied in various environments—crosswalks, public transport stations, educational institutions, and more. It has the potential to become an affordable, assistive tool, empowering millions with better mobility and confidence in an otherwise challenging world.

---

## Technical Complexity

**Technical Overview**:  
Beacon integrates multiple advanced technologies:
- **YOLOv7** for real-time object detection, detecting objects like pedestrians, vehicles, and obstacles with precision.
- **Places365 (ResNet-50)** for scene detection, helping users understand the broader environment (e.g., identifying locations like "park," "street," or "building interior").
- **OCR using Azure Cognitive Services** for reading text in the environment, enabling users to "see" text on road signs, billboards, or papers.
- **Voice Interface** using Pyttsx3 and SpeechRecognition, making the system completely hands-free by allowing users to interact via voice commands.

The technical complexity lies in the seamless integration of these technologies. The system efficiently processes video feeds in real-time while handling speech input/output, object/scene recognition, and providing navigation suggestions. The model pipeline uses **multi-threading** to ensure that both object detection and scene recognition run concurrently without lag, enhancing the user experience.

Beacon showcases **deep learning** knowledge, especially in **real-time video processing**, **computer vision**, and **natural language processing**. By using **PyTorch** and **OpenCV** for efficient model execution and **DroidCam** for input feeds, it demonstrates high technical proficiency.

---

## Functionality and Usability

**Ease of Use**:  
The application is designed to run seamlessly with minimal input from the user. Simply by saying "Hey, Beacon," the app begins processing the video feed and providing real-time feedback on the environment. The voice interface ensures that even non-technical users can operate it effortlessly. 

**Smooth Operation**:  
Despite its complexity, Beacon performs its tasks efficiently. Real-time object detection, scene recognition, and text recognition are combined into a single coherent output for the user. Voice feedback is immediate, keeping users informed without overwhelming them. The app has been tested to run on multiple devices using camera feeds through DroidCam, ensuring high compatibility with mobile systems.

**User Journey**:  
1. **Wake the assistant** by saying "Hey, Beacon."
2. **Receive object detection updates** about nearby obstacles or important objects.
3. **Understand your environment** with scene recognition that tells you if you are in a park, on a street, or indoors.
4. **Get text information** about signs or boards in your environment using OCR.
5. **Navigate safely** with voice-guided directions, ensuring that users can understand what’s around them without visual input.

---

## Collaboration and Learning

**Collaboration**:  
As a solo project, **Beacon** reflects a high level of individual exploration, experimentation, and problem-solving. The journey involved mastering the integration of advanced AI models and tools, along with extensive research into accessibility and usability for visually impaired individuals.

**Learning**:  
During the development of Beacon, I explored areas of AI that were previously unfamiliar, including advanced real-time processing techniques, multi-model pipelines, and Android deployment. Working with PyTorch, OpenCV, and integrating Azure Cognitive Services OCR into the project provided hands-on experience with cutting-edge AI frameworks. The deployment challenges on mobile platforms via Buildozer and testing with DroidCam further expanded my skill set in mobile development and AI system optimization.

---

## Conclusion

Beacon is not just a proof-of-concept but a functional prototype that pushes the boundaries of assistive technology. By combining cutting-edge AI techniques with a focus on usability, Beacon addresses real-world challenges faced by visually impaired individuals. With further development and deployment, Beacon could become a transformative tool for millions.

