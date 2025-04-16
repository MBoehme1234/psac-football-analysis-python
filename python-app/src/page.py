from flask import Flask, request, jsonify, send_file, Response
from flask_cors import CORS
import cv2
import os
import time
from dotenv import load_dotenv
from werkzeug.utils import secure_filename
import logging
import gc
import torch
from threading import Thread, Lock
from queue import Queue
import uuid
import json
import base64
from ultralytics import YOLO

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('app.log')
    ]
)

logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Load YOLO model
model_path = os.path.join(os.path.dirname(__file__), 'yolov8n.pt')
if not os.path.exists(model_path):
    print(f"Model not found at {model_path}")
    raise FileNotFoundError("YOLO model file not found. Please ensure yolov8n.pt is in the same directory as page.py")
model = YOLO(model_path)

# Define allowed file extensions
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'jpg', 'jpeg', 'png'}

def allowed_file(filename):
    """Check if the file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Initialize task queue and results storage with thread-safe locks
task_queue = Queue()
processing_results = {}
results_lock = Lock()

def process_frame(frame):
    """Process a single frame with YOLO model"""
    try:
        results = model(frame, verbose=False)
        result = results[0]
        annotated_frame = frame.copy()
        
        # Draw boxes if they exist
        if hasattr(result, 'boxes') and result.boxes is not None:
            boxes = result.boxes
            for box in boxes:
                # Get box coordinates and convert to numpy
                coords = box.xyxy[0].cpu().numpy()
                # Get confidence and class
                conf = float(box.conf[0].cpu().numpy())
                cls = int(box.cls[0].cpu().numpy())
                
                # Draw rectangle
                x1, y1, x2, y2 = map(int, coords)
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Add label with class name and confidence
                label = f'{model.names[cls]} {conf:.2f}'
                cv2.putText(annotated_frame, label, (x1, y1 - 10), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return annotated_frame
    except Exception as e:
        logging.error(f"Error processing frame: {str(e)}")
        return frame

def process_video(video_path, task_id):
    """Process video frame by frame and store results"""
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise Exception("Failed to open video file")

        # Get video properties
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        logging.info(f"Processing video: {frame_width}x{frame_height} @ {fps}fps, {total_frames} frames")

        # Send initial metadata
        with results_lock:
            if task_id not in processing_results:
                processing_results[task_id] = {'events': []}
            processing_results[task_id]['events'].append({
                'type': 'metadata',
                'frame_width': frame_width,
                'frame_height': frame_height,
                'fps': fps,
                'total_frames': total_frames
            })
            logging.info(f"Sent metadata for task {task_id}")

        # Process frames in batches
        frame_count = 0
        batch_size = 30  # Process 30 frames at a time
        current_batch = []
        failed_frames = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            try:
                # Process frame
                processed_frame = process_frame(frame)
                
                # Convert frame to base64
                _, buffer = cv2.imencode('.jpg', processed_frame)
                frame_base64 = base64.b64encode(buffer).decode('utf-8')
                current_batch.append(frame_base64)

                # Update progress
                frame_count += 1
                progress = int((frame_count / total_frames) * 100)
                
                # Send batch when it reaches batch_size or at the end
                if len(current_batch) >= batch_size or frame_count == total_frames:
                    with results_lock:
                        if task_id not in processing_results:
                            processing_results[task_id] = {'events': []}
                        processing_results[task_id]['events'].append({
                            'type': 'batch',
                            'frames': current_batch,
                            'progress': progress,
                            'current_frame': frame_count,
                            'total_frames': total_frames
                        })
                        logging.info(f"Sent batch of {len(current_batch)} frames for task {task_id}. Progress: {progress}%")
                    current_batch = []  # Clear the batch
                    
                    # Force garbage collection after each batch
                    gc.collect()
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None

            except Exception as e:
                logging.error(f"Error processing frame {frame_count}: {str(e)}")
                failed_frames.append(frame_count)

        cap.release()

        # Send completion event
        with results_lock:
            if task_id not in processing_results:
                processing_results[task_id] = {'events': []}
            processing_results[task_id]['events'].append({
                'type': 'complete',
                'total_frames': frame_count,
                'failed_frames': failed_frames
            })
            logging.info(f"Processing complete for task {task_id}. Total frames: {frame_count}, Failed frames: {len(failed_frames)}")

        # Clean up the uploaded file
        try:
            os.remove(video_path)
        except Exception as e:
            logging.error(f"Error removing temporary file: {str(e)}")

    except Exception as e:
        logging.error(f"Error in process_video: {str(e)}")
        with results_lock:
            if task_id not in processing_results:
                processing_results[task_id] = {'events': []}
            processing_results[task_id]['events'].append({
                'type': 'error',
                'error': str(e)
            })

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            file.save(filepath)
            
            # Generate a unique task ID
            task_id = str(uuid.uuid4())
            
            # Initialize task entry
            with results_lock:
                processing_results[task_id] = {'events': []}
            
            # Start processing in a background thread
            thread = Thread(target=process_video, args=(filepath, task_id))
            thread.daemon = True
            thread.start()
            
            return jsonify({
                'message': 'Processing started',
                'task_id': task_id
            })
        else:
            return jsonify({'error': 'File type not allowed'}), 400

    except Exception as e:
        logging.error(f"Error in upload_file: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/events/<task_id>')
def event_stream(task_id):
    """SSE endpoint for real-time status updates"""
    def generate():
        last_event_id = 0
        retry_count = 0
        max_retries = 3
        last_activity = time.time()
        timeout = 60  # 60 seconds timeout
        
        logging.info(f"Starting SSE stream for task {task_id}")
        
        while True:
            try:
                current_time = time.time()
                if current_time - last_activity > timeout:
                    logger.warning(f"Connection timeout for task {task_id}")
                    yield f"data: {json.dumps({'type': 'error', 'message': 'Connection timeout'})}\n\n"
                    break

                with results_lock:
                    if task_id not in processing_results:
                        yield f"data: {json.dumps({'type': 'error', 'message': 'Task not found'})}\n\n"
                        break
                    
                    result = processing_results[task_id]
                    events = result.get('events', [])
                    
                    # Send only new events
                    for event in events[last_event_id:]:
                        # Ensure the event is properly formatted
                        event_data = json.dumps(event)
                        yield f"data: {event_data}\n\n"
                        last_event_id += 1
                        last_activity = current_time
                        
                        # Log event details
                        if event['type'] == 'batch':
                            logging.info(f"Sending batch event for task {task_id} with {len(event['frames'])} frames")
                        elif event['type'] in ['complete', 'error']:
                            logging.info(f"Sending {event['type']} event for task {task_id}")
                        
                        # Clean up completed events to free memory
                        if event['type'] in ['batch', 'complete', 'error']:
                            events.remove(event)
                    
                    # If processing is complete or failed, close the stream
                    if events and events[-1]['type'] in ['complete', 'error']:
                        break
                
                # Send a keep-alive comment every 30 seconds
                if current_time - last_activity > 30:
                    yield ": keep-alive\n\n"
                    last_activity = current_time
                
                time.sleep(0.1)  # Reduced sleep time to be more responsive
                
            except Exception as e:
                logger.error(f"Error in event stream: {str(e)}")
                retry_count += 1
                if retry_count >= max_retries:
                    yield f"data: {json.dumps({'type': 'error', 'message': 'Connection error'})}\n\n"
                    break
                time.sleep(1)  # Wait before retrying
        
        logging.info(f"SSE stream ended for task {task_id}")
    
    response = Response(generate(), mimetype='text/event-stream')
    response.headers['Cache-Control'] = 'no-cache, no-transform'
    response.headers['Connection'] = 'keep-alive'
    response.headers['X-Accel-Buffering'] = 'no'
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    response.headers['Access-Control-Allow-Methods'] = 'GET, OPTIONS'
    return response

@app.route('/uploads/<path:filename>')
@app.route('/uploads/<path:filename>/')
def serve_file(filename):
    """Serve processed video files"""
    print(f"[INFO] Serving file: {filename} from {UPLOAD_FOLDER}")
    try:
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        if not os.path.exists(file_path):
            print(f"[ERROR] File not found: {file_path}")
            return jsonify({'error': 'File not found'}), 404
            
        return send_file(
            file_path,
            mimetype='video/mp4',
            as_attachment=False,
            conditional=True
        )
    except Exception as e:
        print(f"[ERROR] Error serving file: {str(e)}")
        return jsonify({'error': str(e)}), 404

def process_image(image_path, max_size=640):
    """Process image with memory optimization"""
    try:
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("Failed to read image")

        # Resize image if too large
        height, width = img.shape[:2]
        if max(height, width) > max_size:
            scale = max_size / max(height, width)
            new_width = int(width * scale)
            new_height = int(height * scale)
            img = cv2.resize(img, (new_width, new_height))
            logger.info(f"Resized image from {width}x{height} to {new_width}x{new_height}")

        # Convert to RGB for YOLO
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Run YOLO detection
        results = model(img_rgb, conf=0.25, iou=0.45)
        
        # Process results
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Get box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # Draw box
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Add label
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                label = f'{model.names[cls]} {conf:.2f}'
                cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Clean up box data
                del box
            del boxes
        del results

        # Save processed image
        output_path = os.path.join(UPLOAD_FOLDER, 'processed_' + os.path.basename(image_path))
        cv2.imwrite(output_path, img)
        
        # Clean up memory
        del img
        del img_rgb
        gc.collect()
        
        # Additional memory cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return output_path
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        raise

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'})

@app.route('/detect', methods=['POST', 'OPTIONS'])
@app.route('/detect/', methods=['POST', 'OPTIONS'])
def detect():
    if request.method == 'OPTIONS':
        # Handle preflight request
        response = jsonify({'message': 'OK'})
        origin = request.headers.get('Origin')
        if origin in ["http://localhost:3000", "https://psac-football-analysis.vercel.app", "https://*.vercel.app"]:
            response.headers.add('Access-Control-Allow-Origin', origin)
            response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Accept,Origin')
            response.headers.add('Access-Control-Allow-Methods', 'POST,OPTIONS')
            response.headers.add('Access-Control-Allow-Credentials', 'true')
        return response

    try:
        print("[INFO] Received request to /detect endpoint")
        start_time = time.time()
        data = request.get_json()
        
        if not data or 'videoUrl' not in data:
            print("[ERROR] No video URL provided")
            return jsonify({
                'success': False,
                'error': 'No video URL provided'
            }), 400

        video_url = data['videoUrl']
        print(f"[INFO] Received video URL: {video_url}")

        # Extract filename from URL
        filename = os.path.basename(video_url)
        input_path = os.path.join(UPLOAD_FOLDER, filename)
        print(f"[INFO] Looking for file at: {input_path}")

        if not os.path.exists(input_path):
            print(f"[ERROR] File not found: {input_path}")
            return jsonify({
                'success': False,
                'error': f'Video file not found: {filename}'
            }), 404

        # Generate a unique task ID
        task_id = str(uuid.uuid4())
        
        # Initialize task entry
        with results_lock:
            processing_results[task_id] = {'events': []}
        
        # Start processing in a background thread
        thread = Thread(target=process_video, args=(input_path, task_id))
        thread.daemon = True
        thread.start()
        
        # Return the task ID to the frontend
        response = jsonify({
            'success': True,
            'task_id': task_id,
            'message': 'Processing started'
        })
        
        # Add CORS headers
        origin = request.headers.get('Origin')
        if origin in ["http://localhost:3000", "https://psac-football-analysis.vercel.app", "https://*.vercel.app"]:
            response.headers.add('Access-Control-Allow-Origin', origin)
            response.headers.add('Access-Control-Allow-Credentials', 'true')
        
        return response

    except Exception as e:
        print(f"[ERROR] Error processing video: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))