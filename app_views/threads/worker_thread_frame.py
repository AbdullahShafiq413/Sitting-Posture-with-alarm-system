import time
from PyQt5 import QtCore
from app_controllers.utils.frame_helper import *
import cv2
import numpy as np
import pyttsx3
try:
    import winsound   # Windows beep
except ImportError:
    winsound = None
from google import genai
from google.genai import types

'''Thread class for handling the received frames
'''

class WorkerThreadFrame(QtCore.QThread):
    update_camera = QtCore.pyqtSignal(object, object, object, object, object)

    def __init__(self, model, view):
        # Use super() to call __init__() methods in the parent classes
        super(WorkerThreadFrame, self).__init__()
        self.model = model
        self.view = view
        self.inference_model = model.inference_model
        self.slider_brightness = view.slider_brightness
        self.button_rotate = view.button_rotate
        self.slider_contrast = view.slider_contrast
        # Place the camera object in the WorkThread
        self.frame = None

        # Always use laptop camera (index 0)
        self.id = 0
        self.camera = cv2.VideoCapture(self.id, cv2.CAP_DSHOW)

        # set video format to mjpg to compress the frames to increase fps
        self.camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        # set frame resolution
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
        # The boolean variable to break the while loop in self.run() method
        self.running = True
        self.bad_posture_label = "sitting_bad"
        self.bad_seconds_threshold = 30  # seconds
        self.bad_posture_start_time = None
        self.alert_active = False

        # --- Text-to-speech engine ---
        self.tts_engine = pyttsx3.init()
        self.tts_engine.setProperty("rate", 170)
        self.tts_engine.setProperty("volume", 1.0)

    def frame_has_bad_posture(self, results):
        """
        Returns True if current YOLO results contain a 'sitting_bad' detection.
        Assumes YOLOv5-style results (results.xyxy[0], results.names).
        """
        try:
            for *xyxy, conf, cls in results.xyxy[0]:
                label = results.names[int(cls)]
                if label == self.bad_posture_label and float(conf) > 0.5:
                    return True
        except Exception as e:
            print("Error parsing results in frame_has_bad_posture:", e)
        return False
    
    def send_image_to_gemini(self, image_path):
        """Send image to Gemini API for posture analysis and feedback."""
        
        # Set the API key directly in your code
        api_key = "AIzaSyDqkOgDorzZz3NBx3JouwIXwkKCiejKiRY"  # Replace with your actual API key
        
        # Initialize the Gemini client with the API key
        client = genai.Client(api_key=api_key)

        with open(image_path, 'rb') as f:
            image_bytes = f.read()

        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[
                types.Part.from_bytes(
                    data=image_bytes,
                    mime_type='image/jpeg',
                ),
                'What is wrong with this person\'s posture, and provide a suggestion for improvement.'
                'Just give 1 line on how the posture is bad and 1 line suggestion to improve it. Do not give long Paragraphs '
            ]
        )
        
        return response.text

    def trigger_posture_alert(self):
        """Capture image, send to Gemini, and trigger alert with TTS."""
        print("⚠️  Bad posture for 30+ seconds! Triggering alert...")

        # Capture the frame (image) at the moment of bad posture
        image_path = "bad_posture_image.jpg"
        cv2.imwrite(image_path, self.frame)  # Save the captured frame to a file

        # Send image to Gemini API for analysis
        gemini_response = self.send_image_to_gemini(image_path)

        # Beep (Windows)
        if winsound is not None:
            try:
                winsound.Beep(1000, 800)  # frequency, duration in ms
            except Exception as e:
                print("Beep failed:", e)

        # Voice (Text-to-Speech)
        try:
            self.tts_engine.say(gemini_response)  # Use the response from Gemini as TTS
            self.tts_engine.runAndWait()
        except Exception as e:
            print("TTS failed:", e)

    def run(self):
        frame_count = 0
        start_time = time.time()
        fps = 0
        while self.running:
            # read one frame
            b, self.frame = self.camera.read()
            if b:
                frame_count += 1
                elapsed_time = time.time() - start_time
                if elapsed_time >= 1:
                    fps = frame_count / elapsed_time
                    frame_count = 0
                    start_time = time.time()
            # change brightness based on slider value
            self.frame = change_brightness(self.frame, self.slider_brightness.value() / 100)
            # change contrast based on slider value
            self.frame = change_contrast(self.frame, self.slider_contrast.value() / 100)
            self.check_orientation()
            self.check_rotation()
            # predict using inference_models
            results = self.inference_model.predict(self.frame)
            now = time.time()
            has_bad_posture = self.frame_has_bad_posture(results)

            if has_bad_posture:
                if self.bad_posture_start_time is None:
                    self.bad_posture_start_time = now
                    self.alert_active = False
                else:
                    elapsed_bad = now - self.bad_posture_start_time
                    if elapsed_bad >= self.bad_seconds_threshold and not self.alert_active:
                        self.trigger_posture_alert()
                        self.alert_active = True
            else:
                # posture good again → reset
                self.bad_posture_start_time = None
                self.alert_active = False

            # --- New: on-screen warning while alert active ---
            if self.alert_active and self.frame is not None:
                # Ensure we have a proper, contiguous uint8 image for OpenCV drawing
                if not isinstance(self.frame, np.ndarray):
                    frame_for_draw = np.array(self.frame)
                else:
                    frame_for_draw = self.frame

                frame_for_draw = np.ascontiguousarray(frame_for_draw)

                msg = "WARNING: Slouching for 30+ seconds! Sit up straight."
                cv2.putText(
                    frame_for_draw,
                    msg,
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),  # red
                    2,
                    cv2.LINE_AA,
                )

                # Put it back so the GUI sees the updated frame
                self.frame = frame_for_draw

            # Send updated frame + results to the GUI
            self.update_camera.emit(self.model, self.view, self.frame, fps, results)

    def stop(self):
        # terminate the while loop in self.run() method
        self.running = False
        self.camera.release()
        cv2.destroyAllWindows()

    def check_rotation(self):
        if self.model.frame_rotation == 90:
            self.frame = np.rot90(self.frame, -1, (0, 1))
        elif self.model.frame_rotation == 180:
            self.frame = np.rot90(self.frame, -2, (0, 1))
        elif self.model.frame_rotation == 270:
            self.frame = np.rot90(self.frame, -3, (0, 1))

    def check_orientation(self):
        if self.model.frame_orientation_vertical == 1:
            self.frame = np.flipud(self.frame)
        if self.model.frame_orientation_horizontal == 1:
            self.frame = np.fliplr(self.frame)
