import cv2
import numpy as np
import time
import threading
import queue


class VideoProcessor:
    """
    Processes video frames for the ASD model with optimizations for 
    low-end devices and real-time performance.
    """
    
    def __init__(self, config=None):
        """
        Initialize the video processor with configuration parameters.
        
        Args:
            config (dict): Configuration parameters including:
                - input_source (int/str): Camera index or video file path (default: 0)
                - frame_width (int): Target frame width (default: 640)
                - frame_height (int): Target frame height (default: 480)
                - fps (int): Target frame rate (default: 30)
                - max_queue_size (int): Maximum size of frame queue (default: 10)
                - use_threading (bool): Whether to use threaded processing (default: True)
        """
        # Default configuration
        self.config = {
            'input_source': 0,  # Default camera
            'frame_width': 640,
            'frame_height': 480,
            'fps': 30,
            'max_queue_size': 10,
            'use_threading': True,
            'process_every_n_frames': 2,  # Process every 2nd frame by default for performance
            'face_detection_interval': 3  # Detect faces every 3 frames
        }
        
        # Update with provided config
        if config:
            self.config.update(config)
        
        # Initialize variables
        self.capture = None
        self.is_running = False
        self.frame_count = 0
        self.last_fps_update = time.time()
        self.current_fps = 0.0
        self.processed_frames = 0
        self.frame_times = []
        
        # Frame queues for threading
        self.raw_frame_queue = queue.Queue(maxsize=self.config['max_queue_size'])
        self.processed_frame_queue = queue.Queue(maxsize=self.config['max_queue_size'])
        
        # Locks for thread safety
        self.capture_lock = threading.Lock()
        
        # Performance monitoring
        self.processing_times = []
        
        print(f"VideoProcessor initialized with config: {self.config}")
    
    def start(self):
        """
        Start video capture and processing.
        """
        # Initialize video capture
        try:
            self.capture = cv2.VideoCapture(self.config['input_source'])
            
            # Configure camera properties
            self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.config['frame_width'])
            self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config['frame_height'])
            self.capture.set(cv2.CAP_PROP_FPS, self.config['fps'])
            
            # Optimize buffer size for lower latency
            self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            # Set MJPG codec for better performance if possible
            self.capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            
            if not self.capture.isOpened():
                raise Exception("Failed to open video capture device.")
                
            # Get actual frame dimensions (may differ from requested)
            actual_width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = self.capture.get(cv2.CAP_PROP_FPS)
            
            print(f"Camera initialized: {actual_width}x{actual_height} @ {actual_fps:.1f}fps")
            
            # Start capture thread if threading is enabled
            self.is_running = True
            
            if self.config['use_threading']:
                self.capture_thread = threading.Thread(target=self._capture_loop)
                self.capture_thread.daemon = True
                self.capture_thread.start()
                
                # Start processing thread
                self.process_thread = threading.Thread(target=self._process_loop)
                self.process_thread.daemon = True
                self.process_thread.start()
                
                print("Video capture and processing threads started")
            
            return True
            
        except Exception as e:
            print(f"Error starting video capture: {e}")
            if self.capture:
                self.capture.release()
                self.capture = None
            return False
    
    def stop(self):
        """
        Stop video capture and processing.
        """
        self.is_running = False
        
        # Wait for threads to finish
        if self.config['use_threading'] and hasattr(self, 'capture_thread'):
            self.capture_thread.join(timeout=1.0)
            self.process_thread.join(timeout=1.0)
        
        # Release camera
        if self.capture:
            self.capture.release()
            self.capture = None
            
        print("Video processor stopped")
    
    def get_frame(self):
        """
        Get the latest processed frame.
        
        Returns:
            tuple: (frame, metadata) or (None, None) if no frame is available
        """
        if not self.config['use_threading']:
            # Direct capture mode
            return self._capture_and_process_frame()
        
        # Threaded mode - get from queue
        try:
            if not self.processed_frame_queue.empty():
                return self.processed_frame_queue.get_nowait()
            return None, None
        except queue.Empty:
            return None, None
    
    def _capture_loop(self):
        """
        Main thread for continuous frame capture.
        """
        frame_interval = 1.0 / self.config['fps']
        skip_counter = 0
        
        while self.is_running:
            loop_start = time.time()
            
            # Skip some frames if processing is falling behind
            if self.raw_frame_queue.qsize() >= self.config['max_queue_size'] // 2:
                skip_counter += 1
                if skip_counter <= 2:  # Skip up to 2 frames when backed up
                    time.sleep(0.001)  # Minimal sleep
                    continue
                else:
                    # Reset skip counter after skipping some frames
                    skip_counter = 0
            
            # Capture frame
            with self.capture_lock:
                ret, frame = self.capture.read()
            
            if not ret:
                print("Failed to capture frame")
                time.sleep(0.01)
                continue
            
            # Add to queue, dropping frames if full
            try:
                if self.raw_frame_queue.full():
                    try:
                        # Remove oldest frame
                        self.raw_frame_queue.get_nowait()
                    except queue.Empty:
                        pass
                
                self.raw_frame_queue.put((frame, {'timestamp': time.time()}), block=False)
                
            except queue.Full:
                # Skip if queue is full
                pass
            
            # Update FPS calculation
            self.frame_count += 1
            elapsed = time.time() - self.last_fps_update
            if elapsed >= 1.0:
                self.current_fps = self.frame_count / elapsed
                self.frame_count = 0
                self.last_fps_update = time.time()
            
            # Sleep to maintain target frame rate
            elapsed = time.time() - loop_start
            if elapsed < frame_interval:
                time.sleep(frame_interval - elapsed)
    
    def _process_loop(self):
        """
        Thread for processing captured frames.
        """
        while self.is_running:
            try:
                # Get frame from queue
                if self.raw_frame_queue.empty():
                    time.sleep(0.001)  # Minimal sleep
                    continue
                
                frame_data = self.raw_frame_queue.get(timeout=0.1)
                if frame_data is None:
                    continue
                    
                frame, metadata = frame_data
                
                # Process the frame
                start_time = time.time()
                processed_frame, updated_metadata = self._process_frame(frame, metadata)
                process_time = time.time() - start_time
                
                # Track processing performance
                self.processing_times.append(process_time)
                if len(self.processing_times) > 30:
                    self.processing_times.pop(0)
                
                # Update metadata with processing info
                if updated_metadata:
                    updated_metadata['process_time'] = process_time
                    updated_metadata['avg_process_time'] = sum(self.processing_times) / len(self.processing_times)
                
                # Add to output queue, dropping frames if full
                if self.processed_frame_queue.full():
                    try:
                        self.processed_frame_queue.get_nowait()
                    except queue.Empty:
                        pass
                
                self.processed_frame_queue.put((processed_frame, updated_metadata), block=False)
                self.processed_frames += 1
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error in frame processing: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(0.01)
    
    def _capture_and_process_frame(self):
        """
        Capture and process a single frame (for non-threaded mode).
        
        Returns:
            tuple: (processed_frame, metadata) or (None, None) if capture fails
        """
        with self.capture_lock:
            if not self.capture or not self.capture.isOpened():
                return None, None
                
            ret, frame = self.capture.read()
            
        if not ret:
            return None, None
        
        metadata = {'timestamp': time.time()}
        return self._process_frame(frame, metadata)
    
    def _process_frame(self, frame, metadata):
        """
        Process a captured frame for use with the ASD model.
        
        Args:
            frame: The raw captured frame
            metadata: Frame metadata
            
        Returns:
            tuple: (processed_frame, updated_metadata)
        """
        try:
            # Increment frame counter
            self.frame_count += 1
            
            # Resize frame if needed for consistent processing
            h, w = frame.shape[:2]
            target_h, target_w = self.config['frame_height'], self.config['frame_width']
            
            if h != target_h or w != target_w:
                frame = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)
            
            # Basic preprocessing
            # 1. Apply histogram equalization to improve contrast in varying lighting
            if len(frame.shape) == 3:  # Color image
                # Convert to YUV and equalize the Y channel
                yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
                yuv[:,:,0] = cv2.equalizeHist(yuv[:,:,0])
                frame_eq = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
                
                # Blend with original for more natural look
                alpha = 0.3  # Adjust blending factor
                frame = cv2.addWeighted(frame, 1-alpha, frame_eq, alpha, 0)
            
            # 2. Optional: Noise reduction for better feature extraction
            if self.frame_count % 2 == 0:  # Only apply to some frames to save processing
                frame = cv2.fastNlMeansDenoisingColored(frame, None, 5, 5, 7, 21)
            
            # Update metadata
            metadata['processed'] = True
            metadata['frame_count'] = self.frame_count
            
            return frame, metadata
            
        except Exception as e:
            print(f"Error processing frame: {e}")
            return frame, metadata  # Return original frame if processing fails
    
    def get_performance_stats(self):
        """
        Get video processing performance statistics.
        
        Returns:
            dict: Performance metrics
        """
        avg_process_time = 0
        if self.processing_times:
            avg_process_time = sum(self.processing_times) / len(self.processing_times)
        
        return {
            'fps': self.current_fps,
            'avg_process_time': avg_process_time,
            'processed_frames': self.processed_frames,
            'queue_size': self.raw_frame_queue.qsize() if hasattr(self, 'raw_frame_queue') else 0
        }


# Example usage if run directly
if __name__ == "__main__":
    # Simple test to demonstrate the VideoProcessor
    processor = VideoProcessor({
        'input_source': 0,  # Default camera
        'frame_width': 640,
        'frame_height': 480,
        'fps': 30
    })
    
    if processor.start():
        try:
            # Create a window
            cv2.namedWindow("Video Processor Test", cv2.WINDOW_NORMAL)
            
            while True:
                # Get processed frame
                frame, metadata = processor.get_frame()
                
                if frame is None:
                    time.sleep(0.01)
                    continue
                
                # Add performance stats to the frame
                stats = processor.get_performance_stats()
                cv2.putText(frame, f"FPS: {stats['fps']:.1f}", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(frame, f"Process time: {stats['avg_process_time']*1000:.1f}ms", (10, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
                # Display the frame
                cv2.imshow("Video Processor Test", frame)
                
                # Exit on 'q' key
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        except KeyboardInterrupt:
            print("Interrupted by user")
        finally:
            processor.stop()
            cv2.destroyAllWindows()
    else:
        print("Failed to start video processor")