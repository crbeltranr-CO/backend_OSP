import os
import cv2
import time
import shutil
import logging
import argparse
import threading
import RPi.GPIO as GPIO
from tflite_support.task import core
from tflite_support.task import vision
from tflite_support.task import processor
from flask import Flask, Response, render_template, send_file

"""
    Author: Daniel Montenegro
    Email: Pending
    Modify_by: Cristian Beltran
    Date_Creation: 2023-10-01 (modificar)
    Date_Modification: 2025-05-17
    Purpose:
    This script implements a real-time object detection system using TensorFlow Lite on a Raspberry Pi.
    It captures video frames from a camera, detects objects in the frames, and triggers actions based on the detections.
    The system can save video clips of detected events, manage storage capacity, and provide a web interface for live streaming and event retrieval.
    Requirements:
    - TensorFlow Lite
    - OpenCV
    - Flask
    Version: 1.2
"""

class Buzzer:
    """
    A class that manages a buzzer connected to a Raspberry Pi pin using PWM.
    This class allows controlling the buzzer, emitting sounds with a specific frequency 
    and adjustable duty cycle, and it also has the capability to perform automatic 
    blinking (turning on and off) with a specified number of cycles.
    """
    def __init__(self, pin=12, frequency=5000, duty_cycle=50):
        """
        Initializes the `Buzzer` class by setting up the pin the buzzer is connected to, the frequency 
        at which the buzzer operates (in Hertz), the duty cycle (percentage of time it's on during each cycle),
        and sets up the initial configuration of the buzzer.

        Parameters:
        - pin: GPIO pin the buzzer is connected to (default is pin 12).
        - frequency: Frequency at which the buzzer emits sound, in Hertz (default is 5000 Hz).
        - duty_cycle: The duty cycle of the buzzer, percentage of time it stays on during each cycle (default is 50%).
        """

        self.pin = pin # GPIO pin the buzzer is connected to
        GPIO.setmode(GPIO.BCM) # Sets the GPIO pin numbering mode (BCM refers to GPIO numbering)
        GPIO.setup(self.pin, GPIO.OUT) # Sets the pin as an output
        self.frequency = frequency # Frequency of the buzzer
        self.pwm = GPIO.PWM(self.pin, self.frequency) # Creates a PWM instance on the pin with the defined frequency
        self.duty_cycle = duty_cycle # Sets the buzzer's duty cycle (percentage of time it's on)
        self.active = False # Boolean indicating if the buzzer is currently active

    def auto_stop(self, cycles=3, duration=0.1):
        
        """
        Automatically blinks the buzzer: turns it on and off for a specified number of cycles,
        with adjustable duration for each on-off cycle.

        Parameters:
        - cycles: The number of on-off cycles the buzzer will perform (default is 3).
        - duration: Duration in seconds for each on and off phase (default is 0.1 seconds).
        """

        if not self.active: # If the buzzer is not currently active
            self.active = True # Marks the buzzer as active
            for _ in range(cycles): # Performs the on-off cycles
                self.start() # Starts the buzzer
                time.sleep(duration) # Waits for the specified duration
                self.stop() # Stops the buzzer
                time.sleep(duration) # Waits for the specified duration
            self.active = False # Marks the buzzer as inactive
    
    def start(self): 
        """
        Starts the buzzer by activating PWM with the defined duty cycle. 
        The buzzer will emit sound at the frequency specified by `self.frequency`
        and with the duty cycle defined by `self.duty_cycle`.
        """
        self.pwm.start(self.duty_cycle) # Starts the PWM with the defined duty cycle
    
    def stop(self):
        """
        Stops the buzzer and the associated PWM signal, effectively turning off the buzzer.
        """
        self.pwm.stop() # Stops the PWM signal, turning off the buzzer

class ObjectDetector:
    """
    A class that provides object detection capabilities using a TensorFlow Lite model.
    This class allows detecting objects in an image by utilizing an EfficientDet Lite model, 
    and filtering the results based on the specified score threshold and category name allowlist.
    """

    def __init__(self, model_name="efficientdet_lite0.tflite", num_threads=4, score_threshold=0.3, max_results=1, category_name_allowlist=["person"]):
        """
        Initializes the `ObjectDetector` class by loading the specified TensorFlow Lite model and 
        setting up the detection options such as the number of threads, score threshold, maximum results, 
        and allowed object categories.
        
        Parameters:
        - model_name: The file name of the TensorFlow Lite object detection model (default is "efficientdet_lite0.tflite").
        - num_threads: The number of threads to use for inference (default is 4).
        - score_threshold: The score threshold to filter detections. Detections with a score lower than this value will be ignored (default is 0.3).
        - max_results: The maximum number of results to return (default is 1).
        - category_name_allowlist: A list of object categories to allow in the detection results (default is ["person"]).
        """

        # Set up the base options for the model, including the number of threads and whether to use a Coral accelerator.
        base_options = core.BaseOptions(file_name=model_name, use_coral=False, num_threads=num_threads)

        # Set up the detection options, including the maximum number of results, score threshold, and allowed categories.
        detection_options = processor.DetectionOptions(max_results=max_results, score_threshold=score_threshold, category_name_allowlist=category_name_allowlist)

        # Combine the base and detection options to create the full object detection configuration.
        options = vision.ObjectDetectorOptions(base_options=base_options, detection_options=detection_options)

        # Create the object detector from the configured options.
        self.detector = vision.ObjectDetector.create_from_options(options)

    def detections(self, image):
        """
        Performs object detection on the provided image. The image is first converted from BGR (OpenCV format) 
        to RGB, as required by the TensorFlow Lite model. The function returns the detected objects in the image.

        Parameters:
        - image: The input image (in BGR format) on which to perform object detection.

        Returns:
        - A list of detected objects in the image, including information such as their bounding box coordinates and detection scores.
        """

        # Convert the image from BGR (OpenCV format) to RGB.
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Perform object detection and return the detections.
        return self.detector.detect(vision.TensorImage.create_from_array(rgb_image)).detections

class Camera:
    """
    A class to manage video capture from a camera (webcam) using OpenCV.
    This class allows you to configure the camera resolution and capture frames from it.
    """

    def __init__(self, frame_width=1280, frame_height=720, camera_number=0):
        """
        Initializes the `Camera` class by opening the video stream from the specified camera.
        It also sets the desired resolution (frame width and height) for the video stream.
        
        Parameters:
        - frame_width: The width of the video frame (default is 1280 pixels).
        - frame_height: The height of the video frame (default is 720 pixels).
        - camera_number: The camera number to use (default is 0, which typically refers to the default webcam).
        """

        # Open the video capture stream for the specified camera number.
        self.video_capture = cv2.VideoCapture(camera_number)

        # Set the desired frame width and height for the video stream.
        self.video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
        self.video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

    def frame(self):
        """
        Captures a single frame from the video stream.

        Returns:
        - frame: The captured video frame as a numpy array.
        """

        # Read a frame from the video stream.
        _, frame = self.video_capture.read()

        # Return the captured frame.
        return frame

class RealTimeObjectDetection:
    """
    A class to manage real-time object detection, video streaming, event recording, and security breach detection.
    It uses a camera to capture video frames, performs object detection on each frame, and triggers actions 
    such as saving event footage, controlling LEDs, and activating a buzzer when a security breach occurs.
    """

    def __init__(self, frame_width=1280, frame_height=720, camera_number=0, model_name="efficientdet_lite0.tflite", num_threads=4, score_threshold=0.3, max_results=1, category_name_allowlist=["person"], 
                 folder_name="events", storage_capacity=32, led_pines=[(13, 19, 26), (21, 20, 16)], pin_buzzer=12, frequency=5000, duty_cycle=50, fps_frame_count= 30, safe_zone=((0, 0), (1280, 720))):
        """
        Initializes the `RealTimeObjectDetection` class by setting up the camera, object detector, storage manager,
        LED control, buzzer, and other parameters necessary for real-time monitoring and event handling.
        
        Parameters:
        - frame_width: Width of the video frames (default is 1280).
        - frame_height: Height of the video frames (default is 720).
        - camera_number: The camera number to use (default is 0).
        - model_name: TensorFlow Lite model for object detection (default is "efficientdet_lite0.tflite").
        - num_threads: Number of threads to use for object detection (default is 4).
        - score_threshold: The threshold to filter object detections by confidence score (default is 0.3).
        - max_results: The maximum number of objects to detect (default is 1).
        - category_name_allowlist: List of allowed object categories (default is ["person"]).
        - folder_name: Folder name to save event videos (default is "events").
        - storage_capacity: The maximum storage capacity (default is 32GB).
        - led_pines: GPIO pins for controlling RGB LEDs (default is [(13, 19, 26), (21, 20, 16)]).
        - pin_buzzer: GPIO pin for the buzzer (default is 12).
        - frequency: Frequency of the buzzer sound (default is 5000 Hz).
        - duty_cycle: Duty cycle of the buzzer (default is 50%).
        - fps_frame_count: Number of frames used to calculate FPS (default is 30).
        - safe_zone: Coordinates for defining a safe zone in the frame (default is the full frame from (0, 0) to (1280, 720)).
        """
        
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.camera = Camera(frame_width, frame_height, camera_number)
        self.frame = self.camera.frame()
        self.object_detector = ObjectDetector(model_name, num_threads, score_threshold, max_results, category_name_allowlist)
        self.folder_name = folder_name
        self.storage_manager = StorageManager(folder_name, storage_capacity)
        self.storage_manager.supervise_folder_capacity()
        self.leds_rgb = LEDSRGB(led_pines)
        self.buzzer = Buzzer(pin_buzzer, frequency, duty_cycle)
        self.safe_zone_start, self.safe_zone_end = safe_zone
        self.fps_frame_count = fps_frame_count
        self.last_detection_timestamp = None
        self.frame_buffer = []
        self.frame_times = []
        self.output = {}
        self.events = 0
        self.fps = 24

    def guard(self, min_video_duration=1, max_video_duration=60, max_detection_delay=10, event_check_interval=10, safe_zone=False):
        """
        Main method for continuously monitoring the video feed for security breaches.
        When a breach is detected, it records the event, triggers the buzzer and LED, and saves the footage.
        
        Parameters:
        - min_video_duration: Minimum duration to record an event (default is 1 second).
        - max_video_duration: Maximum duration to record an event (default is 60 seconds).
        - max_detection_delay: Maximum delay to wait before stopping recording (default is 10 seconds).
        - event_check_interval: Interval to check storage capacity and manage files (default is 10).
        - safe_zone: Flag to highlight the safe zone in the frame (default is False).
        """
        
        try:
            self.buzzer.auto_stop() # Stops the buzzer automatically if it's active
            self.leds_rgb.set_color(["off", "green"]) # Sets LEDs to green to indicate no security breach

            while self.isOpened():
                security_breach, time_localtime = self.process_frame((0, 0, 255), 1, 2, cv2.FONT_HERSHEY_SIMPLEX, safe_zone)

                if security_breach: # If a security breach is detected

                    if not self.frame_buffer: # If no video is being recorded
                        self.output["file_name"] = time.strftime("%B%d_%Hhr_%Mmin%Ssec", time_localtime)
                        self.output["day"], self.output["hours"], self.output["mins"] = self.output["file_name"].split("_")
                        self.output["path"] = os.path.join(self.folder_name, self.output["day"], self.output["hours"], f"{self.output['file_name']}.mp4")
                    
                    elif len(self.frame_buffer) == int(self.fps): # After capturing 1 second of footage
                        buzzer_thread = threading.Thread(target=self.buzzer.auto_stop)
                        buzzer_thread.start() # Start the buzzer in a separate thread
                        self.leds_rgb.red() # Set LEDs to red to indicate security breach

                    self.last_detection_timestamp = time.time() # Record the time of the detection
                    self.frame_buffer.append(self.frame) # Add the current frame to the buffer

                else: # If no security breach is detected
                    if self.last_detection_timestamp and ((time.time() - self.last_detection_timestamp) >= max_detection_delay):
                        if len(self.frame_buffer) >= self.fps*min_video_duration: # If the event is long enough
                            self.save_frame_buffer(self.output["path"], event_check_interval)
                        self.leds_rgb.set_color(["off", "green"]) # Set LEDs back to green
                        self.last_detection_timestamp = None # Reset the last detection timestamp
                        self.frame_buffer = [] # Clear the frame buffer
                        self.output = {} # Reset the output metadata

                    elif len(self.frame_buffer) >= self.fps*max_video_duration: # If the event exceeds the maximum video duration
                        self.save_frame_buffer(self.output["path"], event_check_interval)

        except Exception as e:
            logging.error(e, exc_info=True) # Log any errors that occur
            GPIO.cleanup() # Clean up GPIO pins
            self.close() # Release the camera
            os._exit(0) # Exit the program
    
    def save_frame_buffer(self, path, event_check_interval=10):
        """
        Saves the buffered frames as a video file to the specified path.
        
        Parameters:
        - path: The path where the video file will be saved.
        - event_check_interval: Interval to check storage capacity and manage files (default is 10).
        """

        output_seconds = int(len(self.frame_buffer)/self.fps) # Calculate the length of the event in seconds
        os.makedirs(os.path.dirname(path), exist_ok=True) # Create the directory if it doesn't exist
        out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"avc1"), self.fps, (self.frame_width, self.frame_height)) 
        logging.warning(f"EVENT: {output_seconds} seconds {path}") # Log the event's duration and path
        for frame in self.frame_buffer: 
            out.write(frame) # Write each frame to the video file
        out.release() # Release the video writer
        self.events += 1 # Increment the event counter

        if self.events % event_check_interval == 0: # Periodically check the storage capacity
            storage_thread = threading.Thread(target=self.storage_manager.supervise_folder_capacity)
            storage_thread.start()
    
    def _safe_zone_invasion(self, rect_start, rect_end):
        """
        Checks if a detected object has entered the defined safe zone.
        
        Parameters:
        - rect_start: The starting point of the bounding box (top-left corner).
        - rect_end: The ending point of the bounding box (bottom-right corner).
        
        Returns:
        - True if the object is inside the safe zone, False otherwise.
        """

        if self.safe_zone_start[0] > rect_end[0] or self.safe_zone_end[0] < rect_start[0]:
            return False
        if self.safe_zone_start[1] > rect_end[1] or self.safe_zone_end[1] < rect_start[1]:
            return False
        return True

    def process_frame(self, color=(0, 0, 255), font_size=1, font_thickness=2, font=cv2.FONT_HERSHEY_SIMPLEX, safe_zone=False):
        """
        Processes a frame from the camera, performs object detection, and checks for security breaches.
        
        Parameters:
        - color: The color of the bounding box and text (default is red).
        - font_size: The font size for the text (default is 1).
        - font_thickness: The thickness of the font (default is 2).
        - font: The font style for the text (default is cv2.FONT_HERSHEY_SIMPLEX).
        - safe_zone: Whether to highlight the safe zone in the frame (default is False).
        
        Returns:
        - security_breach: Boolean indicating whether a security breach was detected.
        - time_localtime: The local time at which the frame was processed.
        """

        security_breach = False
        start_time = time.time()
        frame = self.camera.frame()
        time_localtime = time.localtime()
        detections = self.object_detector.detections(frame)
        for detection in detections:
            box = detection.bounding_box
            rect_start = (box.origin_x, box.origin_y)
            rect_end = (box.origin_x+box.width, box.origin_y+box.height)
            category_name = detection.categories[0].category_name
            text_position = (7+box.origin_x, 21+box.origin_y)
            cv2.putText(frame, category_name, text_position, font, font_size, color, font_thickness)
            cv2.rectangle(frame, rect_start, rect_end, color, font_thickness)
            security_breach = self._safe_zone_invasion(rect_start, rect_end)
        cv2.putText(frame, time.strftime("%B%d/%Y %H:%M:%S", time_localtime), (21, 42), cv2.FONT_HERSHEY_SIMPLEX, font_size, color, font_thickness)
        if safe_zone:
            cv2.rectangle(frame, self.safe_zone_start, self.safe_zone_end, (0, 255, 255), font_thickness)
        self.frame = frame
        self.frame_times.append(time.time() - start_time)
        if self.fps_frame_count == len(self.frame_times):
            average_frame_time = sum(self.frame_times) / len(self.frame_times)
            self.fps = round(1/average_frame_time, 2)
            self.frame_times = []
        return security_breach, time_localtime

    def isOpened(self):
        """
        Checks if the camera is open and ready to capture frames.
        
        Returns:
        - True if the camera is open, False otherwise.
        """
        return self.camera.video_capture.isOpened()
    
    def close(self):
        """
        Releases the camera and closes the video capture stream.
        """
        self.camera.video_capture.release()

class StorageManager:

    def __init__(self, events_folder="events", storage_capacity=32):
        
        self.events_folder = events_folder
        self.storage_capacity = storage_capacity

    @staticmethod
    def folder_size_gb(folder_path):
        total_size_bytes = 0
        for dirpath, _, filenames in os.walk(folder_path):
            for filename in filenames:
                file_path = os.path.join(dirpath, filename)
                total_size_bytes += os.path.getsize(file_path)
        return total_size_bytes / (1024 ** 3)
    
    @staticmethod
    def delete_folder(folder_path):
        folder_size = StorageManager.folder_size_gb(folder_path)
        shutil.rmtree(folder_path)
        logging.warning(f"STORAGE: '{folder_path}' was deleted (-{folder_size:.4f} GB)")
        return folder_size

    def supervise_folder_capacity(self):
        events_folder_size = StorageManager.folder_size_gb(self.events_folder)
        logging.info(f"STORAGE: '{self.events_folder}' is ({events_folder_size:.4f} GB)")
        while events_folder_size > self.storage_capacity:
            folder_to_delete = os.path.join(self.events_folder, min(os.listdir(self.events_folder)))
            events_folder_size -= StorageManager.delete_folder(folder_to_delete)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder-name", default="events", help="Name of the folder to store events (default: 'events')")
    parser.add_argument("--log-file", default="logfile.log", help="Name of the log file (default: 'logfile.log')")
    parser.add_argument("--reset-events", action="store_true", help="Reset events folder")
    parser.add_argument("--reset-logs", action="store_true", help="Reset log file")
    args = parser.parse_args()
    try:
        log_file = args.log_file
        if args.reset_logs:
            with open(log_file, "w") as file:
                file.write(f"{log_file.upper()}\n")
        logging.basicConfig(filename=log_file, level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", datefmt="%B%d/%Y %H:%M:%S")

        folder_name = args.folder_name
        if args.reset_events:
            StorageManager.delete_folder(folder_name)
        os.makedirs("events", exist_ok=True)

        remote_camera = RealTimeObjectDetection(
            frame_width=1280,
            frame_height=720,
            camera_number=0,
            model_name="efficientdet_lite0.tflite",
            num_threads=4,
            score_threshold=0.5,
            max_results=3, 
            category_name_allowlist=["person", "dog", "cat", "umbrella"],
            folder_name=folder_name,
            storage_capacity=32,
            led_pines=[(13, 19, 26), (16, 20, 21)],
            pin_buzzer=12,
            frequency=5000,
            duty_cycle=50,
            fps_frame_count=30,
            safe_zone=((0, 180), (1280, 720))
        )

        guard_thread = threading.Thread(target=remote_camera.guard, kwargs={
            "min_video_duration": 1,
            "max_video_duration": 60,
            "max_detection_delay": 10,
            "event_check_interval": 10,
            "safe_zone": True
        })
        guard_thread.start()

        app = Flask(__name__)

        def real_time_transmission(duration=300):
            start_time = time.time()
            time_seconds = int(time.time())
            while time_seconds - start_time < duration:
                if time_seconds % 2:
                    cv2.circle(remote_camera.frame, (1238, 21), 12, (0, 255, 0), -1)
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + cv2.imencode(".jpg", remote_camera.frame)[1].tobytes() + b"\r\n")
                time_seconds = int(time.time())
            cv2.circle(remote_camera.frame, (1238, 21), 12, (0, 0, 255), -1)
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + cv2.imencode(".jpg", remote_camera.frame)[1].tobytes() + b"\r\n")

        @app.route("/")
        def stream_video():
            return Response(real_time_transmission(), mimetype="multipart/x-mixed-replace; boundary=frame")
        
        @app.route("/logs/")
        def get_logs():
            with open(log_file, "r") as file:
                return Response(file.read(), mimetype="text/plain")
        
        @app.route("/events/")
        def get_events():
            days = []
            h1 = "EVENTS"
            for day in sorted(os.listdir(folder_name)):
                day_path = os.path.join(folder_name, day)
                day_info = {"date": day, "hours": []}
                for hour in sorted(os.listdir(day_path)):
                    hour_path = os.path.join(day_path, hour)
                    hour_info  = {"time": hour, "videos": []}
                    for video in sorted(os.listdir(hour_path)):
                        video_name = "".join(video.split("_")[1:]).replace(".mp4", "")
                        hour_info["videos"].append({"name": video_name, "path": video})
                    day_info["hours"].append(hour_info )
                days.append(day_info)
            if not days:
                h1 = "NO EVENTS AVAIBLE"
            return render_template("events.html", events=days, h1=h1)
        
        @app.route("/play/<path:video_path>")
        def get_video(video_path):
            video_path = os.path.join(folder_name, video_path)
            if os.path.exists(video_path):
                return send_file(video_path, mimetype="video/mp4")
            else:
                return "Video not found."


        @app.route("/live/")
        def live_video():
            def generate ():
                while True:
                    frame = remote_camera.frame
                    _, buffer = cv2.imencode(".jpg", frame)
                    yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")

            return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")

        app.run(host="0.0.0.0", port=80, threaded=True)   
    except Exception as e:
        logging.error(e, exc_info=True)
        remote_camera.close()
        GPIO.cleanup()
    finally:
        remote_camera.close()
        GPIO.cleanup()