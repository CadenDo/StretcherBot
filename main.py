# Libraries 
import time # for FPS calculation
import cv2 # OpenCV 
import numpy as np # for frame/array type hints
from ultralytics import YOLO # Pose Detection AI Podel

# Global Program Configuration
modelSel: str = "yolo26n-pose.pt" # Ultralytics Model
cameraSelect: int = 0 # Camera Selection (for multiple cameras, default is 0)
confidence: float = 0.5 # arbirtrary value for testing 
resolution: int = (1280, 736) # 720p-ish resolution (YOLO needs both values to be divisible by 32); 0 is width, 1 is height
outputWindowName: str = "Camera + Pose Detection Output"

# Initalizes camera feed and sets inital camera parameters.
def cameraInit(cameraSelect: int) -> cv2.VideoCapture: # paramater is the camera index, returns a cv2.VideoCapture
    # Variable Declaration
    camConfig: cv2.VideoCapture
    
    # Camera Capture, there seems to be a noticable performance difference when using cv2.CAP_DSHOW and not using... 
    # Boot time is defined as model is fully operational and output is showing: Didn't boot within 2 minutes without cv2.CAP_DSHOW, 8.3 second boot time with cv2.CAP_DSHOW  
    camConfig = cv2.VideoCapture(cameraSelect, cv2.CAP_DSHOW)
    print(f"[CameraInit] Camera index {cameraSelect} opened successfully.")
    #camera = cv2.VideoCapture(cameraSelect)

    if not camConfig.isOpened():
        raise RuntimeError( f"Could not open camera index {cameraSelect}.") # f allows string printing with variable. Without it, just prints the variable name.

    camConfig.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
    camConfig.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
    camConfig.set(cv2.CAP_PROP_FPS, 60)
    print("[CameraInit] Camera resolution and FPS set up successfully.")

    # suppose to reduce latancy, TEST LATER...
    camConfig.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    #print("[CameraInit] cameraInit completed")

    return camConfig

def main() -> None:
    # Variable Declaration
    model: YOLO
    mainCamera: cv2.VideoCapture
    prevTime: float
    smoothedFPS: float
    bootTimeStart: float = time.perf_counter()
    bootMeasured: bool = False # boot time lock; False is unlocked, true is locked 
    
    model = YOLO(modelSel) 
    print(f"[Main] Model initialized.")
    
    # Initalize the camera feed and set up the output window.
    mainCamera = cameraInit(cameraSelect)
    cv2.namedWindow(outputWindowName, cv2.WINDOW_NORMAL)
    print(f"[Main] Camera initialized and output window created.")

    # FPS Counter
    prevTime = time.perf_counter()
    smoothedFPS = 0.00

    print("[Main] Camera started. Press Esc key to quit.")

    while True:
        # Variable Declaration
        success: bool
        frame: np.ndarray 
        frameResult: np.ndarray
        annotatedFrame: np.ndarray
        currTime: float
        instantFPS: float
        prevTime: float
        smoothedFPS: float = 0.00
        
        success, frame = mainCamera.read() # outputs an array of the current frame and a boolean for success/failure

        if not success:
            print("[Main] Failed to read a frame from the camera")
            break

        # Run YOLO pose estimation on the OpenCV BGR frame.
        frameResult = model.predict(source=frame, conf=confidence, imgsz=resolution, verbose=False) # change verbose to True to display model performance metrics

        # Draw bounding boxes, keypoints, and skeletons.
        annotatedFrame = frameResult[0].plot()

        # Calculate and display an approximate output FPS.
        currTime = time.perf_counter()
        instantFPS = 1.0 / max(currTime - prevTime, 1e-6)
        prevTime = currTime

        if smoothedFPS == 0.00:
            smoothedFPS = instantFPS
        else:
            smoothedFPS = 0.90 * smoothedFPS + 0.10 * instantFPS
            
        # Prints calculated FPS and annotated video frame to output window
        cv2.putText(annotatedFrame, f"FPS: {smoothedFPS:.2f}", (20, 40), cv2.FONT_HERSHEY_PLAIN, 3.0, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow(outputWindowName, annotatedFrame)
        
        # Prints boot time and then locks
        if not bootMeasured:
            bootTimeEnd = time.perf_counter()
            bootTime = bootTimeEnd - bootTimeStart
            print(f"[Main] Boot time to first output: {bootTime:.2f} second")
            bootMeasured = True
            del bootTimeStart
            del bootTimeEnd
            del bootTime

        if cv2.waitKey(1) & 0xFF == 27: # wait for 1 ms and check if pressed key is Esc Key
            print("[Main] Camera program ended")
            break

    # Terminates window
    mainCamera.release()
    cv2.destroyAllWindows()
    print("[Main] Camera released")

# Allows program to run only when called by "python main.py" and not when imported as a header file
if __name__ == "__main__":
    main()