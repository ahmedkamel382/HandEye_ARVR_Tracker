import io
import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, HTMLResponse
from starlette.responses import StreamingResponse

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# --- API and Model Initialization ---

app = FastAPI(title="Touchless HCI Hand Tracker API", version="2.0")

# Standard 21-point hand connections (index pairs), matching MediaPipe's topology.
# Defined manually because mp.solutions (legacy API) has been removed from
# recent mediapipe releases -- we don't rely on it anywhere in this service.
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),          # index finger
    (5, 9), (9, 10), (10, 11), (11, 12),     # middle finger
    (9, 13), (13, 14), (14, 15), (15, 16),   # ring finger
    (13, 17), (17, 18), (18, 19), (19, 20),  # pinky
    (0, 17),                                 # palm base
]

print("Loading MediaPipe Hand Landmarker model into API memory...")
MODEL_PATH = "hand_landmarker.task"

# Each request here is an independent uploaded frame (not a guaranteed-in-order
# video stream), so IMAGE running mode is the correct choice -- it re-runs full
# palm detection on every call rather than assuming temporal continuity.
_base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
_landmarker_options = mp_vision.HandLandmarkerOptions(
    base_options=_base_options,
    running_mode=mp_vision.RunningMode.IMAGE,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)
model = mp_vision.HandLandmarker.create_from_options(_landmarker_options)


def _decode_image(image_bytes: bytes):
    """Decodes uploaded bytes into a BGR OpenCV image, or None if invalid."""
    np_arr = np.frombuffer(image_bytes, np.uint8)
    return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)


def _run_detection(bgr_img):
    """Runs the Hand Landmarker on a BGR image and returns the raw result."""
    rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_img)
    return model.detect(mp_image)


def draw_landmarks_on_image(bgr_image, detection_result):
    """Draws hand landmarks + handedness label onto a BGR (OpenCV-style) image."""
    hand_landmarks_list = detection_result.hand_landmarks
    handedness_list = detection_result.handedness
    annotated_image = np.copy(bgr_image)
    height, width, _ = annotated_image.shape

    for idx in range(len(hand_landmarks_list)):
        hand_landmarks = hand_landmarks_list[idx]
        handedness = handedness_list[idx]

        points = [(int(lm.x * width), int(lm.y * height)) for lm in hand_landmarks]

        for start_idx, end_idx in HAND_CONNECTIONS:
            cv2.line(annotated_image, points[start_idx], points[end_idx], (255, 255, 255), 2)

        for (x, y) in points:
            cv2.circle(annotated_image, (x, y), 4, (0, 128, 255), -1)
            cv2.circle(annotated_image, (x, y), 4, (0, 0, 0), 1)

        x_coords = [p[0] for p in points]
        y_coords = [p[1] for p in points]
        text_x = min(x_coords)
        text_y = min(y_coords) - 10

        cv2.putText(
            annotated_image,
            f"{handedness[0].category_name}",
            (text_x, text_y),
            cv2.FONT_HERSHEY_DUPLEX,
            1,
            (88, 205, 54),
            1,
            cv2.LINE_AA,
        )

    return annotated_image


# --- REST API Endpoints ---

@app.get("/")
def home():
    """Simple health check endpoint to verify the server is running."""
    return {"message": "Touchless HCI Hand Landmarker API is live!"}


@app.post("/predict/")
async def predict_hand(file: UploadFile = File(...)):
    """
    Receives an uploaded image from the browser, runs the MediaPipe Hand
    Landmarker, and returns the 21 normalized (x, y, z) landmarks per
    detected hand in JSON format, along with handedness.
    """
    try:
        image_bytes = await file.read()
        img = _decode_image(image_bytes)

        if img is None:
            return JSONResponse(status_code=400, content={"error": "Invalid image file format."})

        result = _run_detection(img)

        hands = []
        for hand_landmarks, handedness in zip(result.hand_landmarks, result.handedness):
            hands.append({
                "handedness": handedness[0].category_name,
                "confidence": round(handedness[0].score * 100, 2),
                "landmarks": [
                    {"x": lm.x, "y": lm.y, "z": lm.z} for lm in hand_landmarks
                ],
            })

        return {"hands_detected": len(hands), "hands": hands}

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/predict_and_draw/")
async def predict_and_draw(file: UploadFile = File(...)):
    """
    Standard visual debugging endpoint (useful for Swagger UI testing).
    Returns the frame with hand landmarks drawn on it.
    """
    try:
        image_bytes = await file.read()
        img = _decode_image(image_bytes)

        if img is None:
            return JSONResponse(status_code=400, content={"error": "Invalid image file format."})

        result = _run_detection(img)
        annotated_img = draw_landmarks_on_image(img, result)

        _, encoded_img = cv2.imencode('.jpg', annotated_img)
        return StreamingResponse(io.BytesIO(encoded_img.tobytes()), media_type="image/jpeg")

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


# --- Client-Side Web Interface ---

@app.get("/webcam", response_class=HTMLResponse)
def webcam_interface():
    """
    Serves the advanced Frontend HTML/JS application.
    It accesses the viewer's local webcam and communicates with the /predict/ endpoint.
    """
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Live Hand Tracking (Client-Side)</title>
        <style>
    body { 
        background-color: #111; 
        color: white; 
        font-family: sans-serif; 
        text-align: center; 
    }
    #video-container { 
        position: relative; 
        display: inline-block; 
        margin-top: 20px; 
    }
    /* ONLY mirror the video so it acts like a mirror */
    video { 
        border-radius: 8px; 
        transform: scaleX(-1); 
        background: #000;
    }
    /* Keep the canvas normal so text renders left-to-right */
    canvas { 
        position: absolute; 
        top: 0; 
        left: 0; 
        border-radius: 8px;
    }
    #status { margin-top: 15px; color: #00ff88; font-weight: bold; }
</style>
    </head>
    <body>
        <h2>🖐 Remote Touchless HCI Tracking</h2>
        <p>This securely uses your device's camera and streams frames to the Edge AI server.</p>

        <div id="video-container">
            <video id="video" width="640" height="480" autoplay playsinline></video>
            <canvas id="canvas" width="640" height="480"></canvas>
        </div>

        <p id="status">Waiting for camera permissions...</p>

        <script>
            const video = document.getElementById('video');
            const canvas = document.getElementById('canvas');
            const ctx = canvas.getContext('2d');
            const status = document.getElementById('status');

            // Same 21-point connection topology as the Python backend.
            const HAND_CONNECTIONS = [
                [0, 1], [1, 2], [2, 3], [3, 4],
                [0, 5], [5, 6], [6, 7], [7, 8],
                [5, 9], [9, 10], [10, 11], [11, 12],
                [9, 13], [13, 14], [14, 15], [15, 16],
                [13, 17], [17, 18], [18, 19], [19, 20],
                [0, 17],
            ];

            // A hidden canvas just to take "photos" of the video feed
            const captureCanvas = document.createElement('canvas');
            captureCanvas.width = 640;
            captureCanvas.height = 480;
            const captureCtx = captureCanvas.getContext('2d');

            // 1. Turn on the Viewer's Camera
            async function startCamera() {
                try {
                    const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } });
                    video.srcObject = stream;
                    status.innerText = "Camera active. Streaming data to AI Server...";

                    // Start the infinite processing loop once the video is playing
                    video.onloadeddata = () => processFrame();
                } catch (err) {
                    status.innerText = "Error accessing camera. Please allow permissions.";
                    console.error(err);
                }
            }

            // 2. The Main AI Loop
            function processFrame() {
                if (video.readyState !== video.HAVE_ENOUGH_DATA) {
                    requestAnimationFrame(processFrame);
                    return;
                }

                // Copy the current video frame to the hidden canvas
                captureCtx.drawImage(video, 0, 0, 640, 480);

                // Compress the image to a tiny JPEG to save network bandwidth
                captureCanvas.toBlob(async (blob) => {
                    const formData = new FormData();
                    formData.append('file', blob, 'frame.jpg');

                    try {
                        // Send the image to your FastAPI Python server
                        const response = await fetch('/predict/', {
                            method: 'POST',
                            body: formData
                        });

                        const data = await response.json();

                        // Draw the landmark skeletons on the screen
                        drawHands(data.hands);
                    } catch (err) {
                        console.error("Inference Error:", err);
                    }

                    // Instantly trigger the next frame
                    requestAnimationFrame(processFrame);

                }, 'image/jpeg', 0.6); // 60% JPEG quality is the sweet spot for fast uploads
            }

            // 3. Draw the Hand Landmark Skeletons
function drawHands(hands) {
    // Clear the previous frame's drawing
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (!hands || hands.length === 0) return;

    hands.forEach(hand => {
        // Landmarks come back normalized (0-1). Convert to pixel coords and
        // mirror the X axis so points line up with the CSS-flipped video.
        const points = hand.landmarks.map(lm => ({
            x: canvas.width - (lm.x * canvas.width),
            y: lm.y * canvas.height
        }));

        // Skeleton connections
        ctx.strokeStyle = "#00ff88";
        ctx.lineWidth = 2;
        HAND_CONNECTIONS.forEach(([startIdx, endIdx]) => {
            const start = points[startIdx];
            const end = points[endIdx];
            ctx.beginPath();
            ctx.moveTo(start.x, start.y);
            ctx.lineTo(end.x, end.y);
            ctx.stroke();
        });

        // Landmark dots
        ctx.fillStyle = "#00ff88";
        points.forEach(p => {
            ctx.beginPath();
            ctx.arc(p.x, p.y, 4, 0, 2 * Math.PI);
            ctx.fill();
        });

        // Handedness label above the wrist/topmost point
        const minY = Math.min(...points.map(p => p.y));
        const labelX = points[0].x;
        ctx.font = "bold 18px sans-serif";
        ctx.fillStyle = "#111";
        ctx.fillRect(labelX, minY - 30, 140, 25);
        ctx.fillStyle = "#00ff88";
        ctx.fillText(`${hand.handedness} ${hand.confidence}%`, labelX + 5, minY - 12);
    });
}

            // Initialize
            startCamera();
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)