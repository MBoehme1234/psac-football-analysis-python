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
import numpy as np
import subprocess
import tempfile
import shutil

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

# More permissive CORS configuration
CORS(app, 
     resources={r"/*": {
         "origins": "*",
         "methods": ["GET", "POST", "OPTIONS"],
         "allow_headers": ["Content-Type", "Authorization", "X-Requested-With", "Accept", "Cache-Control"],
         "expose_headers": ["Content-Type", "X-SSE-Event"],
         "supports_credentials": True,
         "max_age": 3600,
         "send_wildcard": True
     }})

# Add CORS headers to all responses
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type, Authorization, X-Requested-With, Accept, Cache-Control')
    response.headers.add('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
    response.headers.add('Access-Control-Allow-Credentials', 'true')
    response.headers.add('Access-Control-Max-Age', '3600')
    return response

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
TEMP_FOLDER = os.path.join(UPLOAD_FOLDER, 'temp')

# Ensure both directories exist
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
if not os.path.exists(TEMP_FOLDER):
    os.makedirs(TEMP_FOLDER)

# Configure tempfile to use our directory
tempfile.tempdir = TEMP_FOLDER

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
    """Process video frame by frame and create a new MP4"""
    try:
        # Clean up old temp files before processing
        for f in os.listdir(TEMP_FOLDER):
            try:
                file_path = os.path.join(TEMP_FOLDER, f)
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                logger.warning(f"Error cleaning temp file {f}: {e}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise Exception("Failed to open video file")

        # Get video properties
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        logging.info(f"Processing video: {frame_width}x{frame_height} @ {fps}fps, {total_frames} frames")

        # Process frames and save as images
        frame_count = 0
        failed_frames = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            try:
                # Process frame
                processed_frame = process_frame(frame)
                
                # Save frame as image
                frame_path = os.path.join(TEMP_FOLDER, f'frame_{frame_count:06d}.png')
                success = cv2.imwrite(frame_path, processed_frame)
                
                if not success:
                    logging.error(f"Failed to write frame {frame_count}")
                    failed_frames.append(frame_count)
                elif frame_count % 100 == 0:
                    logging.info(f"Successfully wrote frame {frame_count}")
                
                # Update progress
                frame_count += 1
                progress = int((frame_count / total_frames) * 100)
                
                # Send progress update
                with results_lock:
                    if task_id not in processing_results:
                        processing_results[task_id] = {'events': []}
                    processing_results[task_id]['events'].append({
                        'type': 'progress',
                        'progress': progress,
                        'current_frame': frame_count,
                        'total_frames': total_frames
                    })

            except Exception as e:
                logging.error(f"Error processing frame {frame_count}: {str(e)}")
                failed_frames.append(frame_count)

        # Release video capture
        cap.release()

        if frame_count == 0:
            raise Exception("No frames were processed")

        # Create output video using FFmpeg
        output_filename = f'processed_{task_id}.mp4'
        output_path = os.path.join(UPLOAD_FOLDER, output_filename)
        
        # FFmpeg command to create video from frames
        ffmpeg_cmd = [
            'ffmpeg',
            '-y',  # Overwrite output file if it exists
            '-framerate', str(fps),
            '-i', os.path.join(TEMP_FOLDER, 'frame_%06d.png'),
            '-c:v', 'libx264',  # Use H.264 codec
            '-preset', 'medium',  # Balance between speed and compression
            '-pix_fmt', 'yuv420p',  # Standard pixel format for web playback
            '-movflags', '+faststart',  # Enable fast start for web playback
            output_path
        ]
        
        logging.info(f"Running FFmpeg command: {' '.join(ffmpeg_cmd)}")
        
        # Run FFmpeg
        try:
            result = subprocess.run(
                ffmpeg_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            if result.returncode != 0:
                logging.error(f"FFmpeg error: {result.stderr}")
                raise Exception(f"FFmpeg failed with error: {result.stderr}")
            else:
                logging.info("FFmpeg successfully created video file")
            
        except Exception as e:
            logging.error(f"Error running FFmpeg: {str(e)}")
            raise

        # Verify the output file exists and has size
        if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
            raise Exception("Output video file is empty or does not exist")
        else:
            file_size = os.path.getsize(output_path)
            logging.info(f"Successfully created output video file. Size: {file_size} bytes")

        # Send completion event with video URL
        with results_lock:
            if task_id not in processing_results:
                processing_results[task_id] = {'events': []}
            processing_results[task_id]['events'].append({
                'type': 'complete',
                'video_url': f'/uploads/{output_filename}',
                'total_frames': frame_count,
                'failed_frames': failed_frames
            })
            logging.info(f"Processing complete for task {task_id}. Total frames: {frame_count}, Failed frames: {len(failed_frames)}")

        # Clean up the original uploaded file
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

@app.route('/upload', methods=['POST', 'OPTIONS'])
def upload_file():
    if request.method == 'OPTIONS':
        return '', 204

    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            
            try:
                file.save(filepath)
                logger.info(f"Successfully saved uploaded file: {filepath}")
                
                # Generate a unique task ID
                task_id = str(uuid.uuid4())
                
                # Initialize task entry
                with results_lock:
                    processing_results[task_id] = {'events': []}
                
                # Start processing in a separate thread
                thread = Thread(target=process_video, args=(filepath, task_id))
                thread.daemon = True
                thread.start()
                
                return jsonify({
                    'task_id': task_id,
                    'message': 'File uploaded successfully. Processing started.',
                    'status': 'processing'
                }), 202
                
            except Exception as e:
                logger.error(f"Error saving or processing file: {str(e)}")
                # Clean up the file if it was saved
                if os.path.exists(filepath):
                    try:
                        os.remove(filepath)
                    except Exception as cleanup_error:
                        logger.error(f"Error cleaning up file after failed processing: {cleanup_error}")
                return jsonify({'error': f'Error processing file: {str(e)}'}), 500
        else:
            return jsonify({'error': f'File type not allowed. Allowed types: {", ".join(ALLOWED_EXTENSIONS)}'}), 400
            
    except Exception as e:
        logger.error(f"Unexpected error in upload_file: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

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
    # SSE specific CORS headers
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Credentials'] = 'true'
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
            
        # Get file size
        file_size = os.path.getsize(file_path)
        
        # Get range header
        range_header = request.headers.get('Range', None)
        
        if range_header:
            # Parse range header
            start_bytes = int(range_header.split('=')[1].split('-')[0])
            end_bytes = min(start_bytes + 1024 * 1024, file_size - 1)  # 1MB chunks
            
            # Read file chunk
            with open(file_path, 'rb') as f:
                f.seek(start_bytes)
                data = f.read(end_bytes - start_bytes + 1)
            
            # Create response
            response = Response(
                data,
                206,  # Partial content
                mimetype='video/mp4',
                direct_passthrough=True
            )
            
            # Add headers
            response.headers.add('Content-Range', f'bytes {start_bytes}-{end_bytes}/{file_size}')
            response.headers.add('Accept-Ranges', 'bytes')
            response.headers.add('Content-Length', str(end_bytes - start_bytes + 1))
            response.headers.add('Content-Type', 'video/mp4')
            response.headers.add('Cache-Control', 'no-cache')
            response.headers.add('Connection', 'keep-alive')
            
        else:
            # Full file response
            response = send_file(
                file_path,
                mimetype='video/mp4',
                as_attachment=False,
                conditional=True
            )
            response.headers.add('Accept-Ranges', 'bytes')
            response.headers.add('Content-Length', str(file_size))
            response.headers.add('Cache-Control', 'no-cache')
            response.headers.add('Connection', 'keep-alive')
        
        return response
        
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
    try:
        # Ensure the upload directory exists
        if not os.path.exists(UPLOAD_FOLDER):
            os.makedirs(UPLOAD_FOLDER)
            
        # Clean up any existing temporary files
        temp_files = [f for f in os.listdir(UPLOAD_FOLDER) if f.startswith('processed_')]
        for f in temp_files:
            try:
                os.remove(os.path.join(UPLOAD_FOLDER, f))
            except Exception as e:
                logger.warning(f"Failed to remove temporary file {f}: {e}")
                
        # Configure server to only listen on localhost
        app.run(
            host='127.0.0.1',
            port=5000,
            debug=False,
            use_reloader=False  # Disable reloader when running with nginx
        )
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        raise