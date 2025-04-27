from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent
import asyncio
import logging
import concurrent.futures

# KOKORO / Audio Imports
from kokoro import KPipeline
import tempfile
import os
import time

# Whisper / Recording Imports
from faster_whisper import WhisperModel
import numpy as np
import pyaudio
import threading
import queue

# Apple Silicon optimization imports
import torch

# Global timeout for long operations (e.g., transcription) - Shortened
LONG_OPERATION_TIMEOUT = 120.0 # 2 minutes

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Global Resources ---
# Initialize MCP server
server = Server("Voice Assistant")
# Initialize PyAudio globally
pyaudio_instance = pyaudio.PyAudio()

# Check if MPS (Metal Performance Shaders) is available for Apple Silicon
# This allows using GPU acceleration on M-series Macs
mps_available = torch.backends.mps.is_available()
logger.info(f"MPS (Metal Performance Shaders) available: {mps_available}")

# We'll use CPU device by default since Whisper has sparse tensor 
# compatibility issues with MPS on some operations -> faster-whisper handles this better
device = "cuda" if torch.cuda.is_available() else "cpu" # Prefer CUDA if available
compute_type = "float16" if device == "cuda" else "int8" # Use float16 on GPU, int8 on CPU
logger.info(f"Using device: {device}, compute_type: {compute_type} for Whisper.")

# --- Redirect stdout/stderr during model loading --- 
original_stderr = sys.stderr # Store original stderr

logger.info("Redirecting stdout/stderr during model initialization (this might take a moment)...")
initialization_successful = False
tts_pipeline = None
whisper_model = None
process_pool = None # Initialize ProcessPoolExecutor later

with contextlib.redirect_stdout(original_stderr), contextlib.redirect_stderr(original_stderr):
    try:
        # Initialize KOKORO TTS pipeline
        logger.info("Initializing KOKORO TTS pipeline...")
        tts_pipeline = KPipeline(lang_code='a')  # 'a' is for auto language detection
        logger.info("KOKORO TTS pipeline initialized")

        # Load the Whisper model
        # Using small.int8 as per user suggestion initially
        whisper_model = WhisperModel("small.int8", device=device, compute_type=compute_type)
        logger.info("Whisper model loaded")

        # Warm-up Whisper model
        logger.info("Warming up Whisper model...")
        warmup_audio = np.zeros(int(0.5 * 16000), dtype=np.float32) # 0.5s silence
        _, _ = whisper_model.transcribe(warmup_audio, beam_size=1) # jit kernels & fill cache
        logger.info("Whisper model warmed up.")

        initialization_successful = True # Mark success
        # Initialize ProcessPoolExecutor after successful model load
        process_pool = concurrent.futures.ProcessPoolExecutor(max_workers=2)
    except Exception as e:
        logging.critical(f"FATAL ERROR during model initialization: {e}", exc_info=True)
        # Keep initialization_successful as False

if not initialization_successful:
    logger.critical("Exiting due to model initialization failure.")
    sys.exit(1) # Use a non-zero exit code for failure

# Audio recording parameters - Reduced CHUNK size
CHUNK = 256
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000

# Global variables for the listening process
audio_queue = queue.Queue() # Queue to pass audio data (NumPy array)
is_listening = False
listening_thread = None
# Global TTS output stream
tts_output_stream = None

def get_audio_level(data):
    """Calculate the audio level with proper scaling for 16-bit audio"""
    audio_data = np.frombuffer(data, dtype=np.int16)
    scaled_data = audio_data.astype(np.float32) / 32768.0
    return np.sqrt(np.mean(scaled_data ** 2))

def _record_audio_thread(max_duration, silence_threshold, silence_duration):
    """Internal function to run recording in a thread"""
    global is_listening, pyaudio_instance

def record_audio(max_duration=30, silence_threshold=0.01, silence_duration=2.0):
    """
    Record audio from the microphone, stop on silence, and put the raw audio
    (as a NumPy float32 array) into the global audio_queue.
    """
    global is_listening
    
    p = pyaudio_instance
    
    # Find the first working input device
    device_index = None
    try:
        for i in range(p.get_device_count()):
            try:
                device_info = p.get_device_info_by_index(i)
                if device_info["maxInputChannels"] > 0:
                    logger.info(f"Using input device: {device_info['name']} (Index: {i})")
                    break
            except Exception as e:
                if logger.isEnabledFor(logging.DEBUG): logger.debug(f"Could not open device {i} ({device_info['name']}): {e}")
                continue
    except Exception as e:
        logger.error(f"Error enumerating audio devices: {e}")

    if device_index is None:
        logger.error("No suitable input device found")
        p.terminate()
        # Signal failure by putting None in the queue
        audio_queue.put(None) 
        is_listening = False
        return
    
    # Open audio stream
    try:
        stream = p.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        input_device_index=device_index,
                        frames_per_buffer=CHUNK)
    except Exception as e:
        logger.error(f"Failed to open audio stream on device {device_index}: {e}")
        # Don't terminate the global instance here
        audio_queue.put(None)
        is_listening = False
        return
    
    # Initialize variables for silence detection
    silent_chunks = 0
    frames = []
    is_listening = True
    recording_started_time = time.time()
    
    try:
        while is_listening and (time.time() - recording_started_time < max_duration):
            try:
                data = stream.read(CHUNK)
                frames.append(data)
                audio_level = get_audio_level(data)
                effective_threshold = silence_threshold * (1 + 0.5 * (time.time() - recording_started_time) / max_duration)
                
                if audio_level < effective_threshold:
                    silent_chunks += 1
                    if silent_chunks * CHUNK / RATE >= silence_duration:
                        logger.info(f"Silence detected (level: {audio_level:.6f} < threshold: {effective_threshold:.6f})")
                        break
                else:
                    # Conditional logging
                    if logger.isEnabledFor(logging.DEBUG): logger.debug(f"Sound detected (level: {audio_level:.6f} >= threshold: {effective_threshold:.6f})")
                    silent_chunks = 0  # Reset silence counter on sound
                    
            except IOError as e:
                logger.error(f"IOError during audio recording: {e}")
                break
    except Exception as e:
        logger.error(f"Error during audio recording: {e}", exc_info=True)
    
    try:
        stream.stop_stream()
        stream.close()
        # Don't terminate the global pyaudio instance here
        # p.terminate() 
    except Exception as e:
        logger.error(f"Error closing audio stream: {e}")
    is_listening = False
    
    # Process recorded frames into a NumPy array
    if frames:
        # Combine frames and convert to float32, scaled to [-1, 1]
        audio_data = b''.join(frames)
        audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
        logger.info(f"Recorded {len(audio_np)/RATE:.2f} seconds of audio.")
        audio_queue.put(audio_np) # Put the NumPy array in the queue
    else:
        logger.info("No audio frames recorded.")
        audio_queue.put(None) # Indicate no audio was saved

def listen(max_duration=30, silence_duration=2.0, silence_threshold=0.01):
    """
    Listen for audio input and return the transcribed text.
    This function starts a new recording thread if needed and waits for the result.
    """
    global is_listening, listening_thread
    
    # PyAudio is initialized globally, no need to check here explicitly
    
    # Stop any existing listening process cleanly
    if is_listening and listening_thread and listening_thread.is_alive():
        logger.info("An existing listening process is active. Stopping it first.")
        is_listening = False
        listening_thread.join(timeout=1)
        if listening_thread.is_alive():
            logger.warning("Existing listening thread did not terminate after timeout.")
    
    # Clear the queue of any stale data
    while not audio_queue.empty():
        _ = audio_queue.get() # Discard stale audio data (NumPy arrays)
        if logger.isEnabledFor(logging.DEBUG): logger.debug("Clearing stale audio data from queue.")
    
    logger.info(f"Starting listen process (max_duration={max_duration}s, silence_duration={silence_duration}s, silence_threshold={silence_threshold})")
    # Start recording in a separate thread
    listening_thread = threading.Thread(
        target=_record_audio_thread, # Target the internal thread function
        kwargs={
            'max_duration': max_duration,
            'silence_duration': silence_duration,
            'silence_threshold': silence_threshold,
        },
        daemon=True # Allow main thread to exit even if this thread is running
    )
    listening_thread.start()
    
    # Wait for the recording thread to put a result (file path or None) in the queue
    audio_np = None
    try:
        # Get the NumPy array (or None) from the queue
        audio_np = await asyncio.get_event_loop().run_in_executor(
            None, lambda: audio_queue.get(timeout=max_duration + 5)) 
        logger.info(f"Received from audio queue: {type(audio_np)}")
    except queue.Empty:
        logger.warning("Listening timed out waiting for audio data from recording thread.")
    
    # Ensure the thread is not marked as listening anymore, regardless of outcome
    is_listening = False 
    # Attempt to join the thread to ensure resources are released
    if listening_thread and listening_thread.is_alive(): 
         if logger.isEnabledFor(logging.DEBUG): logger.debug("Waiting for recording thread to finish...")
         listening_thread.join(timeout=1) 
         if listening_thread.is_alive(): 
              logger.warning("Recording thread did not terminate after timeout.") 
    
    if audio_np is None or len(audio_np) == 0:
        logger.warning("No audio data received from the recording process.")
        return "Error: No speech recorded or audio data is empty."
    
    # Transcribe the audio using faster-whisper in the process pool
    logger.info(f"Transcribing {len(audio_np)/RATE:.2f}s of audio using faster-whisper...")
    transcribed_text = ""
    try:
        # Use the device-optimized model with a timeout via ProcessPoolExecutor
        logger.info("Starting transcription...")
        loop = asyncio.get_event_loop()

        # Define the transcribe function call more explicitly for the executor
        def transcribe_sync(audio_data):
            # faster-whisper returns an iterator of Segment objects and an info object
            segments, info = whisper_model.transcribe(audio_data, beam_size=5) 
            logger.info(f"Detected language '{info.language}' with probability {info.language_probability}")
            # Concatenate segments into a single string
            full_text = "".join(segment.text for segment in segments)
            return full_text
        
        transcribe_task = asyncio.get_event_loop().run_in_executor(
            process_pool, transcribe_sync, audio_np # Pass numpy array
        )
        
        try:
            transcribed_text = await asyncio.wait_for(transcribe_task, timeout=LONG_OPERATION_TIMEOUT)
             logger.info(f"Transcription successful: '{transcribed_text}'")
             return transcribed_text
        except asyncio.TimeoutError:
             logger.error(f"Transcription timed out after {LONG_OPERATION_TIMEOUT} seconds.")
             return "Error: Transcription timed out."
        except Exception as e:
             logger.error(f"Error transcribing audio: {e}", exc_info=True)
             return f"Error: Failed to transcribe audio ({type(e).__name__})."
    except Exception as e:
        logger.error(f"Error transcribing audio: {e}", exc_info=True)
        return f"Error: Failed to transcribe audio ({type(e).__name__})."

# Function implementation for conversation turns
async def conversation_turn(system_message: str, max_listen_duration: int = 30) -> str:
    # ... (rest of the function remains unchanged)

# --- TTS Stream Management ---
def ensure_tts_stream(rate):
    """Opens or reconfigures the global TTS stream if needed."""
    global tts_output_stream, pyaudio_instance
    if tts_output_stream:
        # Check if the rate matches, if not, close and reopen
        # Note: PyAudio stream info access might be platform-dependent or limited
        # For simplicity, we'll assume recreation is needed if rate changes.
        # A more robust check would involve querying stream parameters if possible.
        # If the stream exists but we can't verify the rate, we assume it's okay.
        # A simple heuristic: if rate != current rate used, recreate.
        # Let's store the last used rate. This requires a global or class variable.
        # For now, we'll just reopen if the requested rate differs from a default (e.g., 24000)
        # or if the stream is closed.
        if not tts_output_stream.is_active(): # Or check a stored rate variable
             try:
                 tts_output_stream.close()
             except Exception: pass
             tts_output_stream = None

    if not tts_output_stream:
        logger.info(f"Opening new TTS output stream at {rate} Hz")
        try:
            tts_output_stream = pyaudio_instance.open(format=pyaudio.paFloat32,
                                                      channels=1,
                                                      rate=rate,
                                                      output=True,
                                                      frames_per_buffer=CHUNK) # Use smaller buffer
        except Exception as e:
            logger.error(f"Failed to open TTS output stream: {e}")
            tts_output_stream = None # Ensure it's None on failure
    return tts_output_stream

# Streaming TTS speak implementation
async def speak(text: str, voice: str = "af_bella", speed: float = 1.0) -> str:
    """
    Convert text to speech by streaming audio to a single PyAudio output stream.
    This minimizes latency and eliminates gaps. On macOS, playback rate is set to speed*24000 Hz.
    """
    logger.info(f"Streaming speak request (speed={speed}): {text[:100]}...")
    # Prepare TTS generator
    generator = tts_pipeline(text, voice)
    # Determine output sample rate
    base_rate = 24000
    output_rate = int(base_rate * speed)
    # Ensure TTS stream is open and configured
    tts_output_stream = ensure_tts_stream(output_rate)
    if not tts_output_stream:
        return "Error: could not open audio output stream."

    # Stream audio chunks
    total_chars = len(text)
    stream_error = False
    try:
        for i, (gold, prompt_state, audio) in enumerate(generator):
            if audio is None or not tts_output_stream: # Check stream validity
                continue
            # Ensure float32 array
            if hasattr(audio, 'numpy'):
                chunk = audio.numpy().astype(np.float32)
            elif isinstance(audio, np.ndarray):
                chunk = audio.astype(np.float32)
            else:
                chunk = np.array(audio, dtype=np.float32)
            
            if tts_output_stream.is_active():
                 tts_output_stream.write(chunk.tobytes())
                 if logger.isEnabledFor(logging.DEBUG): logger.debug(f"Streamed chunk {i+1}")
            else:
                 logger.error("TTS output stream is not active. Stopping playback.")
                 stream_error = True
                 break # Stop trying to write if stream is inactive
    except Exception as e:
        logger.error(f"Error during streaming playback: {e}", exc_info=True)
        stream_error = True
        return f"Error during playback: {e}"

    # Clean up
    confirmation = f"Streamed TTS of {total_chars} characters with voice {voice} at {speed}x speed."
    logger.info(confirmation)
    return confirmation

async def serve():
    # Create server options
    options = server.create_initialization_options()
    
    # Initialize TTS stream with default rate before starting server
    ensure_tts_stream(rate=24000) 

    # Using the stdio_server context manager for proper stdio handling
    async with stdio_server() as (read_stream, write_stream):
        logger.info("Voice Assistant MCP Server connected to stdio streams")
        # Run the server with explicit streams
        # Consider raise_exceptions=False for production robustness as suggested
        await server.run(read_stream, write_stream, options, raise_exceptions=True)

def cleanup_resources():
    """Gracefully shut down resources."""
    logger.info("Cleaning up resources...")
    if process_pool:
        logger.info("Shutting down process pool...")
        process_pool.shutdown(wait=True)
    if tts_output_stream:
        logger.info("Closing TTS output stream...")
        tts_output_stream.close()
    if pyaudio_instance:
        logger.info("Terminating PyAudio instance...")
        pyaudio_instance.terminate()
    logger.info("Resource cleanup complete.")

if __name__ == "__main__":
    # logger.info("Script entry point reached (__name__ == '__main__').") # Less noisy
    
    try:
        # Run the server using asyncio
        asyncio.run(serve())
    except KeyboardInterrupt:
        logger.info("Server stopped by KeyboardInterrupt (Ctrl+C).")
    except Exception as e:
        logging.critical(f"FATAL ERROR in MCP server loop: {e}", exc_info=True)
        sys.exit(f"MCP server loop crashed: {e}")
    finally:
        cleanup_resources()
        logger.info("MCP server process ending.") 