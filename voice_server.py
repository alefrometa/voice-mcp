from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent
import asyncio
import logging
import re # Added for sentence splitting
import subprocess # Added for non-blocking playback

# KOKORO / Audio Imports
from kokoro import KPipeline
import soundfile as sf
import tempfile
import os
import time

# Whisper / Recording Imports
import whisper
import numpy as np
import wave
import pyaudio
import threading
import queue

# Additional imports for redirection
import sys
import contextlib

# Apple Silicon optimization imports
import torch

# Global timeout for long operations (e.g., TTS generation, transcription)
LONG_OPERATION_TIMEOUT = 600.0 # 10 minutes

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize our MCP server
server = Server("Voice Assistant")

# Check if MPS (Metal Performance Shaders) is available for Apple Silicon
# This allows using GPU acceleration on M-series Macs
mps_available = torch.backends.mps.is_available()
logger.info(f"MPS (Metal Performance Shaders) available: {mps_available}")

# We'll use CPU device by default since Whisper has sparse tensor 
# compatibility issues with MPS on some operations
device = 'cpu'
logger.info(f"Using device for ML: {device}")

# --- Redirect stdout/stderr during model loading --- 
original_stderr = sys.stderr # Store original stderr

logger.info("Redirecting stdout/stderr during model initialization...")
initialization_successful = False
tts_pipeline = None
whisper_model = None

with contextlib.redirect_stdout(original_stderr), contextlib.redirect_stderr(original_stderr):
    try:
        # Initialize the KOKORO TTS pipeline
        logger.info("Initializing KOKORO TTS pipeline...")
        tts_pipeline = KPipeline(lang_code='a')  # 'a' is for auto language detection
        logger.info("KOKORO TTS pipeline initialized")

        # Load the Whisper model
        # Using CPU for now as Whisper has issues with sparse tensors on MPS
        logger.info(f"Loading Whisper medium model on device: {device}...")
        whisper_model = whisper.load_model("medium", device=device)
        logger.info("Whisper model loaded")
        initialization_successful = True # Mark success
    except Exception as e:
        logging.critical(f"FATAL ERROR during model initialization: {e}", exc_info=True)
        # Keep initialization_successful as False

logger.info(f"Model initialization complete (Success: {initialization_successful}), restoring stdout/stderr.")
# --- Redirection finished --- 

# Exit if initialization failed
if not initialization_successful:
    logger.critical("Exiting due to model initialization failure.")
    sys.exit(1) # Use a non-zero exit code for failure

# Audio recording parameters
CHUNK = 512  # Reduced for more responsive silence detection
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000

# Global variables for the listening process
audio_queue = queue.Queue()
is_listening = False
listening_thread = None

def get_audio_level(data):
    """Calculate the audio level with proper scaling for 16-bit audio"""
    audio_data = np.frombuffer(data, dtype=np.int16)
    # Scale to [-1, 1] range for proper RMS calculation
    scaled_data = audio_data.astype(np.float32) / 32768.0
    return np.sqrt(np.mean(scaled_data ** 2))

def record_audio(max_duration=30, silence_threshold=0.01, silence_duration=2.0):
    """
    Record audio from the microphone with automatic stopping on silence.
    
    Args:
        max_duration: Maximum recording duration in seconds
        silence_threshold: Base threshold for silence detection (0-1 range, relative to full scale)
        silence_duration: Duration of silence to stop recording (seconds)
    """
    global is_listening
    
    p = pyaudio.PyAudio()
    
    # Find the first working input device
    device_index = None
    try:
        for i in range(p.get_device_count()):
            device_info = p.get_device_info_by_index(i)
            if device_info['maxInputChannels'] > 0:
                try:
                    stream = p.open(format=FORMAT,
                                    channels=CHANNELS,
                                    rate=RATE,
                                    input=True,
                                    input_device_index=i,
                                    frames_per_buffer=CHUNK)
                    stream.close()
                    device_index = i
                    logger.info(f"Using input device: {device_info['name']} (Index: {i})")
                    break
                except Exception as e:
                    logger.debug(f"Could not open device {i} ({device_info['name']}): {e}")
                    continue
    except Exception as e:
        logger.error(f"Error enumerating audio devices: {e}")

    if device_index is None:
        logger.error("No suitable input device found")
        p.terminate()
        audio_queue.put(None)
        is_listening = False
        return
    
    try:
        stream = p.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        input_device_index=device_index,
                        frames_per_buffer=CHUNK)
    except Exception as e:
        logger.error(f"Failed to open audio stream on device {device_index}: {e}")
        p.terminate()
        audio_queue.put(None)
        is_listening = False
        return

    # Calibrate background noise level
    logger.info("Calibrating background noise level...")
    background_levels = []
    calibration_chunks = 15 # Sample background noise for ~0.5 seconds
    for _ in range(calibration_chunks):
        try:
            data = stream.read(CHUNK, exception_on_overflow=False)
            level = get_audio_level(data)
            background_levels.append(level)
        except Exception as e:
            logger.error(f"Error during calibration: {e}")
            break
    
    effective_threshold = silence_threshold # Start with the base threshold
    if background_levels:
        # Calculate mean background noise, add a margin (e.g., 1.5x)
        background_noise_estimate = np.mean(background_levels) * 1.5 
        # Set effective threshold to be higher than background noise (e.g., 2.5x background)
        # but never lower than the specified base silence_threshold.
        effective_threshold = max(silence_threshold, background_noise_estimate * 2.5) 
        logger.info(f"Background noise level estimated: {background_noise_estimate:.6f}, Effective silence threshold set to: {effective_threshold:.6f}")
    else:
        logger.warning("Could not calibrate background noise, using base threshold")
    
    logger.info("Recording started... Speak now.")
    
    frames = []
    silent_chunks = 0
    silent_threshold_chunks = int(silence_duration * RATE / CHUNK)
    max_chunks = int(max_duration * RATE / CHUNK)
    
    is_listening = True
    recording_started_time = time.time()
    temp_file_path = None
    
    try:
        while is_listening and (time.time() - recording_started_time < max_duration):
            try:
                data = stream.read(CHUNK, exception_on_overflow=False)
                frames.append(data)
                
                audio_level = get_audio_level(data)
                
                # Check for silence using the *effective* threshold
                if audio_level < effective_threshold:
                    silent_chunks += 1
                    if silent_chunks >= silent_threshold_chunks:
                        logger.info(f"Silence detected (level: {audio_level:.6f} < threshold: {effective_threshold:.6f})")
                        break
                else:
                    # Log when sound is detected above threshold if needed for debugging
                    # logger.debug(f"Sound detected (level: {audio_level:.6f} >= threshold: {effective_threshold:.6f})")
                    silent_chunks = 0  # Reset silence counter on sound
                    
            except IOError as e:
                if e.errno == pyaudio.paInputOverflowed:
                    logger.warning("Input overflowed. Frame dropped.")
                else:
                    logger.error(f"Error reading audio stream: {e}")
                    break
            except Exception as e:
                logger.error(f"Unexpected error during recording loop: {e}")
                break

        if not is_listening:
            logger.info("Recording stopped externally.")
        elif time.time() - recording_started_time >= max_duration:
            logger.info("Maximum recording duration reached.")

    finally:
        try:
            stream.stop_stream()
            stream.close()
        except Exception as e:
            logger.error(f"Error closing audio stream: {e}")
        p.terminate()
        is_listening = False
    
    # Save the recorded audio to a temporary file
    if frames:
        temp_dir = tempfile.gettempdir()
        temp_file_path = os.path.join(temp_dir, f"recording_{int(time.time())}.wav")
        
        try:
            wf = wave.open(temp_file_path, 'wb')
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(p.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(b''.join(frames))
            wf.close()
            logger.info(f"Recording saved to temporary file: {temp_file_path}")
            audio_queue.put(temp_file_path)
        except Exception as e:
            logger.error(f"Error saving audio file: {e}")
            audio_queue.put(None) # Indicate failure
            if temp_file_path and os.path.exists(temp_file_path):
                 try: os.remove(temp_file_path) # Clean up partial file
                 except: pass
    else:
        logger.info("No audio frames recorded.")
        audio_queue.put(None) # Indicate no audio was saved
    
    logger.info("Record audio thread finished")

# Function implementation for listening
async def listen(max_duration: int = 30, silence_duration: float = 2.0, silence_threshold: float = 0.01) -> str:
    """
    Listen for user speech and transcribe it using Whisper.
    
    Args:
        max_duration: Maximum recording duration in seconds (default: 30)
        silence_duration: Duration of silence to stop recording (seconds, default: 2.0)
        silence_threshold: Base threshold for silence detection (0-1 range, default: 0.01). The effective threshold will be adjusted based on background noise.
        
    Returns:
        The transcribed text, or an error message.
    """
    global is_listening, listening_thread
    
    # Ensure PyAudio can be initialized (good check before starting thread)
    try:
        p_test = pyaudio.PyAudio()
        p_test.terminate()
    except Exception as e:
        logger.error(f"PyAudio initialization failed: {e}. Cannot listen.")
        return "Error: Audio system initialization failed."

    # Stop any existing listening process cleanly
    if is_listening and listening_thread and listening_thread.is_alive():
        logger.info("An existing listening process is active. Stopping it first.")
        is_listening = False
        try:
            listening_thread.join(timeout=2) # Give it time to stop
            if listening_thread.is_alive():
                logger.warning("Previous listening thread did not exit cleanly.")
        except Exception as e:
            logger.error(f"Error joining previous listening thread: {e}")
    is_listening = False # Ensure it's reset before starting anew

    # Clear the queue of any stale data
    while not audio_queue.empty():
        stale_file = audio_queue.get()
        logger.debug(f"Clearing stale audio file from queue: {stale_file}")
        if stale_file and os.path.exists(stale_file):
            try: os.remove(stale_file) 
            except: pass
    
    logger.info(f"Starting listen process (max_duration={max_duration}s, silence_duration={silence_duration}s, silence_threshold={silence_threshold})")
    # Start recording in a separate thread
    listening_thread = threading.Thread(
        target=record_audio,
        kwargs={
            'max_duration': max_duration,
            'silence_duration': silence_duration,
            'silence_threshold': silence_threshold
        },
        daemon=True # Allow main thread to exit even if this thread is running
    )
    listening_thread.start()
    
    # Wait for the recording thread to put a result (file path or None) in the queue
    audio_file_path = None
    try:
        # Wait slightly longer than max_duration to allow for processing/saving
        audio_file_path = await asyncio.get_event_loop().run_in_executor(
            None, lambda: audio_queue.get(timeout=max_duration + 5))
        logger.info(f"Received from audio queue: {audio_file_path}")
    except queue.Empty:
        logger.warning("Listening timed out waiting for audio data from recording thread.")
        # Explicitly signal the thread to stop if it hasn't already
        if is_listening:
             is_listening = False 
             # We might not be able to join here reliably if it's stuck
        return "Error: Recording timed out or failed to produce audio data."
    finally:
        # Ensure the thread is not marked as listening anymore, regardless of outcome
        is_listening = False 
        # Attempt to join the thread to ensure resources are released
        if listening_thread and listening_thread.is_alive():
             logger.debug("Waiting for recording thread to finish...")
             listening_thread.join(timeout=1)
             if listening_thread.is_alive():
                  logger.warning("Recording thread did not terminate after timeout.")

    if not audio_file_path:
        logger.warning("No audio file was generated by the recording process.")
        return "Error: No speech recorded or audio could not be saved."
    
    # Transcribe the audio using Whisper with MPS acceleration
    logger.info(f"Transcribing audio file: {audio_file_path}")
    transcribed_text = ""
    try:
        # Ensure the file exists before transcribing
        if not os.path.exists(audio_file_path):
             logger.error(f"Audio file {audio_file_path} not found for transcription.")
             return "Error: Recorded audio file not found."

        # Use the device-optimized model with a timeout
        logger.info("Starting transcription...")
        transcribe_task = asyncio.get_event_loop().run_in_executor(
            None, whisper_model.transcribe, audio_file_path
        )
        
        try:
             result = await asyncio.wait_for(transcribe_task, timeout=LONG_OPERATION_TIMEOUT)
             transcribed_text = result["text"].strip()
             logger.info(f"Transcription successful: '{transcribed_text}'")
             return transcribed_text
        except asyncio.TimeoutError:
             logger.error(f"Transcription timed out after {LONG_OPERATION_TIMEOUT} seconds.")
             # Attempt to cancel the background task if possible
             if not transcribe_task.done():
                 transcribe_task.cancel()
             return "Error: Transcription timed out."
        except Exception as e:
             logger.error(f"Error transcribing audio: {e}", exc_info=True)
             return f"Error: Failed to transcribe audio ({type(e).__name__})."
    except Exception as e:
        logger.error(f"Error transcribing audio: {e}", exc_info=True)
        return f"Error: Failed to transcribe audio ({type(e).__name__})."
    finally:
        # Clean up the temporary audio file after transcription
        if audio_file_path and os.path.exists(audio_file_path):
            try:
                os.remove(audio_file_path)
                logger.info(f"Cleaned up temporary audio file: {audio_file_path}")
            except OSError as e:
                logger.warning(f"Could not remove temp audio file {audio_file_path}: {e}")

# Function implementation for conversation turns
async def conversation_turn(system_message: str, max_listen_duration: int = 30) -> str:
    """
    Perform a complete conversation turn: speak a message and listen for a response.
    
    ***NOTE: This tool SHOULD ALWAYS be used when the assistant needs to speak a message
    and then immediately listen for user voice input. This avoids separate 'speak'
    and 'listen' calls.***

    Args:
        system_message: The message to speak to the user
        max_listen_duration: Maximum duration to listen for a response (default: 30)
        
    Returns:
        The user's response as transcribed text, or an error message.
    """
    # Speak the system message (now async)
    speak_result = await speak(system_message) # Await the async speak function
    logger.info(f"Spoke system message: {speak_result}")
    
    # Listen for the user's response using optimized defaults
    user_response = await listen(max_duration=max_listen_duration, 
                           silence_duration=2.0, 
                           silence_threshold=0.01)
    logger.info(f"User response: {user_response}")
    
    return user_response

# Define the server handlers
@server.list_tools()
async def list_tools() -> list[Tool]:
    """List all available voice assistant tools."""
    return [
        Tool(
            name="speak",
            description="Convert text to speech and play it",
            inputSchema={
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "The text to convert to speech"
                    },
                    "voice": {
                        "type": "string",
                        "description": "The voice to use for speech (default: af_bella)",
                        "default": "af_bella"
                    }
                },
                "required": ["text"]
            }
        ),
        Tool(
            name="listen",
            description="Listen for user speech and transcribe it",
            inputSchema={
                "type": "object",
                "properties": {
                    "max_duration": {
                        "type": "integer", 
                        "description": "Maximum recording duration in seconds",
                        "default": 30
                    },
                    "silence_duration": {
                        "type": "number",
                        "description": "Duration of silence to stop recording (seconds)",
                        "default": 2.0
                    },
                    "silence_threshold": {
                        "type": "number",
                        "description": "Threshold for silence detection (0-1 range)",
                        "default": 0.01
                    }
                }
            }
        ),
        Tool(
            name="list_available_voices",
            description="List all available voices for TTS",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
        Tool(
            name="conversation_turn",
            description="Speak a message and listen for a response in one step",
            inputSchema={
                "type": "object",
                "properties": {
                    "system_message": {
                        "type": "string",
                        "description": "The message to speak to the user"
                    },
                    "max_listen_duration": {
                        "type": "integer",
                        "description": "Maximum duration to listen for a response (seconds)",
                        "default": 30
                    }
                },
                "required": ["system_message"]
            }
        )
    ]

@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Handle tool calls for voice assistant."""
    try:
        if name == "speak":
            text = arguments.get("text")
            voice = arguments.get("voice", "af_bella")
            
            if not text:
                raise ValueError("Missing required parameter: text")
                
            # The speak function now handles chunking internally and is async
            result = await speak(text, voice) # Await the async speak function
            return [TextContent(type="text", text=result)]
            
        elif name == "listen":
            max_duration = arguments.get("max_duration", 30)
            silence_duration = arguments.get("silence_duration", 2.0)
            silence_threshold = arguments.get("silence_threshold", 0.01)
            
            result = await listen(max_duration, silence_duration, silence_threshold)
            return [TextContent(type="text", text=result)]
            
        elif name == "list_available_voices":
            voices = [
                "af_heart", "af_alloy", "af_aoede", "af_bella", "af_jessica", 
                "af_kore", "af_nicole", "af_nova", "af_river", "af_sarah", 
                "af_sky", "am_adam", "am_echo", "am_eric", "am_fenrir", 
                "am_liam", "am_michael", "am_onyx", "am_puck", "bf_alice", 
                "bf_emma", "bf_isabella", "bf_lily", "bm_daniel", "bm_fable", 
                "bm_george", "bm_lewis"
            ]
            
            voice_info = """
            Voice format: [nationality]_[name]
            
            Nationalities:
            - af/am: American Female/Male
            - bf/bm: British Female/Male
            
            Example voices by category:
            - American Female: af_heart, af_bella, af_jessica
            - American Male: am_adam, am_eric, am_michael
            - British Female: bf_alice, bf_emma, bf_lily
            - British Male: bm_daniel, bm_george, bm_lewis
            """
            
            return [TextContent(type="text", text=f"Available voices:\n{', '.join(voices)}\n{voice_info}")]
            
        elif name == "conversation_turn":
            system_message = arguments.get("system_message")
            max_listen_duration = arguments.get("max_listen_duration", 30)
            
            if not system_message:
                raise ValueError("Missing required parameter: system_message")
                
            result = await conversation_turn(system_message, max_listen_duration)
            return [TextContent(type="text", text=result)]
            
        else:
            raise ValueError(f"Unknown tool: {name}")
    except Exception as e:
        logger.error(f"Error in call_tool: {str(e)}", exc_info=True)
        return [TextContent(type="text", text=f"Error: {str(e)}")]

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
    # Open PyAudio stream
    p = pyaudio.PyAudio()
    try:
        stream = p.open(format=pyaudio.paFloat32,
                        channels=1,
                        rate=output_rate,
                        output=True)
    except Exception as e:
        logger.error(f"Failed to open audio output stream: {e}")
        p.terminate()
        return f"Error: could not open audio output."

    # Stream audio chunks
    total_chars = len(text)
    try:
        for i, (gold, prompt_state, audio) in enumerate(generator):
            if audio is None:
                continue
            # Ensure float32 array
            # Convert PyTorch tensor to numpy array if needed
            if hasattr(audio, 'numpy'):  # PyTorch tensor
                chunk = audio.numpy().astype(np.float32)
            elif isinstance(audio, np.ndarray):  # Already numpy array
                chunk = audio.astype(np.float32)
            else:
                chunk = np.array(audio, dtype=np.float32)  # Convert other types
            stream.write(chunk.tobytes())
            logger.debug(f"Streamed chunk {i+1}")
    except Exception as e:
        logger.error(f"Error during streaming playback: {e}", exc_info=True)
        stream.stop_stream()
        stream.close()
        p.terminate()
        return f"Error during playback: {e}"

    # Clean up
    stream.stop_stream()
    stream.close()
    p.terminate()
    confirmation = f"Streamed TTS of {total_chars} characters with voice {voice} at {speed}x speed."
    logger.info(confirmation)
    return confirmation

# Main entry point
async def serve() -> None:
    """Run the voice assistant MCP server using stdio transport."""
    logger.info("Starting Voice Assistant MCP server")
    
    # Create server options
    options = server.create_initialization_options()
    
    # Using the stdio_server context manager for proper stdio handling
    async with stdio_server() as (read_stream, write_stream):
        logger.info("Voice Assistant MCP Server connected to stdio streams")
        # Run the server with explicit streams
        await server.run(read_stream, write_stream, options, raise_exceptions=True)

if __name__ == "__main__":
    logger.info("Script entry point reached (__name__ == '__main__').")
    
    try:
        # Run the server using asyncio
        asyncio.run(serve())
    except KeyboardInterrupt:
        logger.info("Server stopped by KeyboardInterrupt (Ctrl+C).")
    except Exception as e:
        # Log any exception that causes the server loop itself to crash
        logging.critical(f"FATAL ERROR in MCP server loop: {e}", exc_info=True)
        sys.exit(f"MCP server loop crashed: {e}")
    finally:
        logger.info("MCP server process ending.") 