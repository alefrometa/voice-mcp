from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent
import asyncio
import logging
import concurrent.futures
import sys
import contextlib
import os
import tempfile
import time
import queue
import threading
import numpy as np
import pyaudio
from enum import Enum
import torch
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    force=True)  # Force configuration to prevent other loggers from interfering
logger = logging.getLogger(__name__)

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Constants
CHUNK = 256
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
LONG_OPERATION_TIMEOUT = 120.0  # 2 minutes for long operations like transcription

# Tools enum
class VoiceTools(str, Enum):
    SPEAK = "speak"
    LISTEN = "listen"
    CONVERSATION_TURN = "conversation_turn"

# Global state and resources (initialized lazily)
pyaudio_instance = None
tts_pipeline = None  # Legacy, kept for backward compatibility
pipelines = {}  # Map of language codes to KPipeline instances
whisper_model = None
process_pool = None
thread_pool = None
tts_output_stream = None
audio_queue = queue.Queue()
is_listening = False
listening_thread = None
initialization_complete = False
initialization_error = None

def get_audio_level(data):
    """Calculate the audio level with proper scaling for 16-bit audio"""
    audio_data = np.frombuffer(data, dtype=np.int16)
    # Scale to [-1, 1] range for proper RMS calculation
    scaled_data = audio_data.astype(np.float32) / 32768.0
    return np.sqrt(np.mean(scaled_data ** 2))

def record_audio(max_duration=30, silence_threshold=0.01, silence_duration=2.0):
    """
    Record audio from the microphone, stop on silence, and put the raw audio
    (as a NumPy float32 array) into the global audio_queue.
    """
    global is_listening, pyaudio_instance
    
    # Find the first working input device
    device_index = None
    try:
        for i in range(pyaudio_instance.get_device_count()):
            device_info = pyaudio_instance.get_device_info_by_index(i)
            if device_info['maxInputChannels'] > 0:
                try:
                    stream = pyaudio_instance.open(format=FORMAT,
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
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(f"Could not open device {i} ({device_info['name']}): {e}")
                    continue
    except Exception as e:
        logger.error(f"Error enumerating audio devices: {e}")
        audio_queue.put(None)
        is_listening = False
        return

    if device_index is None:
        logger.error("No suitable input device found")
        audio_queue.put(None)
        is_listening = False
        return
    
    # Open audio stream
    try:
        stream = pyaudio_instance.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        input_device_index=device_index,
                        frames_per_buffer=CHUNK)
    except Exception as e:
        logger.error(f"Failed to open audio stream on device {device_index}: {e}")
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
    
    # Initialize variables for silence detection
    silent_chunks = 0
    frames = []
    silent_threshold_chunks = int(silence_duration * RATE / CHUNK)
    
    is_listening = True
    recording_started_time = time.time()
    
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
                    # Conditional logging
                    if logger.isEnabledFor(logging.DEBUG): 
                        logger.debug(f"Sound detected (level: {audio_level:.6f} >= threshold: {effective_threshold:.6f})")
                    silent_chunks = 0  # Reset silence counter on sound
                    
            except IOError as e:
                if hasattr(e, 'errno') and e.errno == pyaudio.paInputOverflowed:
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
            # Don't terminate the global pyaudio instance
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
    
    logger.info("Record audio thread finished")

def get_pipeline_for(language: str) -> 'KPipeline':
    """Return or create a KPipeline for the requested language."""
    # Map ISO-like codes to Kokoro single letters
    lang_map = {"en": "a", "en-gb": "b", "es": "e"}
    code = lang_map.get(language, "a")  # default to American English
    if code not in pipelines:
        # Import KPipeline here to avoid circular import
        from kokoro import KPipeline
        logger.info(f"[TTS] Creating pipeline for lang_code='{code}'")
        pipelines[code] = KPipeline(lang_code=code)
    return pipelines[code]

def ensure_initialized():
    """
    Lazily initialize the voice server resources on first use.
    Returns True if initialization is successful.
    """
    global pyaudio_instance, whisper_model, process_pool, thread_pool, initialization_complete, initialization_error
    
    if initialization_complete:
        return True
    
    if initialization_error:
        logger.error(f"Cannot initialize: previous initialization failed with error: {initialization_error}")
        return False
    
    try:
        logger.info("Initializing voice server resources...")
        
        # Initialize PyAudio
        logger.info("Initializing PyAudio...")
        if not pyaudio_instance:
            pyaudio_instance = pyaudio.PyAudio()
        
        # Only import model-related libraries when needed
        from faster_whisper import WhisperModel
        
        # Check device availability
        device = "cuda" if torch.cuda.is_available() else "cpu"
        compute_type = "float16" if device == "cuda" else "int8"
        logger.info(f"Using device: {device}, compute_type: {compute_type} for Whisper.")
        
        # Initialize Whisper model
        logger.info("Loading Whisper model (small)...")
        whisper_model = WhisperModel("small", device=device, compute_type=compute_type)
        
        # Warm-up Whisper model
        logger.info("Warming up Whisper model...")
        warmup_audio = np.zeros(int(0.5 * 16000), dtype=np.float32)
        segments, _ = whisper_model.transcribe(warmup_audio, beam_size=1)
        # Consume the iterator
        for _ in segments:
            pass
        
        # Initialize KPipelines - we'll create them on-demand
        logger.info("Initializing KOKORO TTS pipeline system...")
        
        # Initialize ProcessPoolExecutor
        logger.info("Initializing process pool...")
        process_pool = concurrent.futures.ProcessPoolExecutor(max_workers=2)
        
        # Initialize ThreadPoolExecutor for Whisper transcription
        logger.info("Initializing thread pool...")
        thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=2)
        
        initialization_complete = True
        logger.info("Voice server resources initialized successfully.")
        return True
        
    except Exception as e:
        initialization_error = str(e)
        logger.error(f"Error initializing voice server resources: {e}", exc_info=True)
        return False

def ensure_tts_stream(rate):
    """Opens or reconfigures the global TTS stream if needed."""
    global tts_output_stream, pyaudio_instance
    
    if not pyaudio_instance:
        logger.error("Cannot create TTS stream: PyAudio not initialized")
        return None
        
    if tts_output_stream:
        # Check if the stream is active, if not, close and reopen
        if not tts_output_stream.is_active():
            try:
                tts_output_stream.close()
            except Exception:
                pass
            tts_output_stream = None

    if not tts_output_stream:
        logger.info(f"Opening new TTS output stream at {rate} Hz")
        try:
            tts_output_stream = pyaudio_instance.open(format=pyaudio.paFloat32,
                                                     channels=1,
                                                     rate=rate,
                                                     output=True,
                                                     frames_per_buffer=CHUNK)
        except Exception as e:
            logger.error(f"Failed to open TTS output stream: {e}")
            tts_output_stream = None
    return tts_output_stream

def cleanup_resources():
    """Gracefully shut down resources."""
    global process_pool, tts_output_stream, pyaudio_instance
    
    logger.info("Cleaning up resources...")
    
    if process_pool:
        logger.info("Shutting down process pool...")
        try:
            process_pool.shutdown(wait=True)
        except Exception as e:
            logger.warning(f"Error shutting down process pool: {e}")
    
    if tts_output_stream:
        logger.info("Closing TTS output stream...")
        try:
            if tts_output_stream.is_active():
                tts_output_stream.stop_stream()
            tts_output_stream.close()
        except Exception as e:
            logger.warning(f"Error closing TTS stream: {e}")
    
    if pyaudio_instance:
        logger.info("Terminating PyAudio instance...")
        try:
            pyaudio_instance.terminate()
        except Exception as e:
            logger.warning(f"Error terminating PyAudio: {e}")
    
    logger.info("Resource cleanup complete.")

def transcribe_sync(audio_data, model):
    """Transcribe audio data using the given whisper model."""
    segments, info = model.transcribe(audio_data, beam_size=5)
    logger.info(f"Detected language '{info.language}' with probability {info.language_probability}")
    full_text = " ".join(segment.text for segment in segments).strip()
    return full_text

class VoiceAssistantServer:
    """Core server logic for the Voice Assistant."""
    
    async def speak(self, text: str, voice: str = "af_bella", language: str = "en") -> str:
        """
        Convert text to speech using the Kokoro TTS engine.
        
        Args:
            text: The text to speak
            voice: The voice to use (if not specified, a voice appropriate for the language will be chosen)
            language: The language code ('en' for English, 'es' for Spanish, etc.)
            
        Returns:
            A confirmation message
        """
        if not ensure_initialized():
            return "Error: Voice server initialization failed."
        
        # 1️⃣ Choose a default voice for the language
        if voice == "af_bella":  # caller didn't override
            voice = {
                "en": "af_bella",
                "en-gb": "bf_emma",
                "es": "em_alex"
            }.get(language, "af_bella")
        
        # 2️⃣ Validate Spanish voices
        if language == "es" and voice not in {"ef_dora", "em_alex", "em_santa"}:
            logger.warning(f"[TTS] Unknown Spanish voice '{voice}', falling back to em_alex")
            voice = "em_alex"
        
        # 3️⃣ Grab language-specific pipeline
        try:
            pipeline = get_pipeline_for(language)
            logger.info(f"[TTS] lang='{language}' → code='{pipeline.lang_code}', voice='{voice}', text='{text[:60]}...'")
            
            # Generate and stream audio
            stream = ensure_tts_stream(24000)
            if not stream:
                return f"Error: Could not open audio output stream."
            
            try:
                # Stream audio chunks
                total_chars = len(text)
                for i, (gold, prompt_state, audio) in enumerate(pipeline(text, voice)):
                    if audio is None or not stream:
                        continue
                    
                    # Ensure float32 array
                    if hasattr(audio, 'numpy'):  # PyTorch tensor
                        chunk = audio.numpy().astype(np.float32)
                    elif isinstance(audio, np.ndarray):  # Already numpy array
                        chunk = audio.astype(np.float32)
                    else:
                        chunk = np.array(audio, dtype=np.float32)  # Convert other types
                    
                    if stream.is_active():
                        stream.write(chunk.tobytes())
                        # Log periodically to show progress
                        if i % 25 == 0 and i > 0:
                            logger.debug(f"[TTS] Streamed {i} chunks")
                    else:
                        logger.error("[TTS] Output stream is not active. Stopping playback.")
                        break
                
                confirmation = f"Spoke {total_chars} chars ({language}) with {voice}"
                logger.info(confirmation)
                return confirmation
            except Exception as e:
                logger.error(f"Error during streaming playback: {e}", exc_info=True)
                return f"Error during playback: {e}"
        except Exception as e:
            logger.exception("[TTS] Failure")
            return f"Error generating speech: {e}"
    
    async def listen(self, max_duration: int = 30, silence_duration: float = 2.0, silence_threshold: float = 0.01) -> str:
        """
        Listen for user speech and transcribe it.
        
        Args:
            max_duration: Maximum recording duration in seconds
            silence_duration: Duration of silence to stop recording
            silence_threshold: Base threshold for silence detection
            
        Returns:
            The transcribed text, or an error message
        """
        global is_listening, listening_thread, audio_queue, thread_pool
        
        if not ensure_initialized():
            return "Error: Voice server initialization failed."
        
        # Stop any existing listening process cleanly
        if is_listening and listening_thread and listening_thread.is_alive():
            logger.info("An existing listening process is active. Stopping it first.")
            is_listening = False
            try:
                listening_thread.join(timeout=2)
                if listening_thread.is_alive():
                    logger.warning("Previous listening thread did not exit cleanly.")
            except Exception as e:
                logger.error(f"Error joining previous listening thread: {e}")
        is_listening = False

        # Clear the queue of any stale data
        while not audio_queue.empty():
            _ = audio_queue.get()
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("Clearing stale audio data from queue.")
        
        logger.info(f"Starting listen process (max_duration={max_duration}s, silence_duration={silence_duration}s)")
        
        # Start recording in a separate thread
        listening_thread = threading.Thread(
            target=record_audio,
            kwargs={
                'max_duration': max_duration,
                'silence_duration': silence_duration,
                'silence_threshold': silence_threshold
            },
            daemon=True
        )
        listening_thread.start()
        
        # Wait for the recording thread to put a result in the queue
        audio_np = None
        try:
            # Wait slightly longer than max_duration
            audio_np = await asyncio.get_event_loop().run_in_executor(
                None, lambda: audio_queue.get(timeout=max_duration + 5))
            if audio_np is not None:
                logger.info(f"Received audio data: {len(audio_np)/RATE:.2f}s")
        except queue.Empty:
            logger.warning("Listening timed out waiting for audio data.")
            if is_listening:
                is_listening = False
            return "Error: Recording timed out."
        finally:
            is_listening = False
            if listening_thread and listening_thread.is_alive():
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug("Waiting for recording thread to finish...")
                listening_thread.join(timeout=1)
                if listening_thread.is_alive():
                    logger.warning("Recording thread did not terminate after timeout.")

        if audio_np is None or len(audio_np) == 0:
            logger.warning("No audio data was received.")
            return "Error: No speech recorded."
        
        # Transcribe the audio using faster-whisper with ThreadPoolExecutor
        logger.info(f"Transcribing {len(audio_np)/RATE:.2f}s of audio...")
        try:
            # Use the ThreadPoolExecutor (which doesn't require pickling)
            logger.info("Starting transcription in thread pool...")
            transcribe_task = asyncio.get_event_loop().run_in_executor(
                thread_pool, transcribe_sync, audio_np, whisper_model
            )
            
            try:
                transcribed_text = await asyncio.wait_for(transcribe_task, timeout=LONG_OPERATION_TIMEOUT)
                logger.info(f"Transcription successful: '{transcribed_text}'")
                return transcribed_text
            except asyncio.TimeoutError:
                logger.error(f"Transcription timed out after {LONG_OPERATION_TIMEOUT} seconds.")
                if not transcribe_task.done():
                    transcribe_task.cancel()
                return "Error: Transcription timed out."
            except Exception as e:
                logger.error(f"Error transcribing audio: {e}", exc_info=True)
                return f"Error: Failed to transcribe audio. {type(e).__name__}"
        except Exception as e:
            logger.error(f"Error transcribing audio: {e}", exc_info=True)
            return f"Error: Failed to transcribe audio. {type(e).__name__}"
    
    async def conversation_turn(self, system_message: str, max_listen_duration: int = 30, language: str = "en") -> str:
        """
        Perform a complete conversation turn: speak a message and listen for a response.
        
        Args:
            system_message: The message to speak to the user
            max_listen_duration: Maximum duration to listen for a response
            language: The language code ('en' for English, 'es' for Spanish, etc.)
            
        Returns:
            The user's response as transcribed text, or an error message
        """
        # Speak the system message in the specified language
        speak_result = await self.speak(system_message, language=language)
        logger.info(f"Spoke system message: {speak_result}")
        
        # Listen for the user's response
        user_response = await self.listen(max_duration=max_listen_duration, 
                            silence_duration=2.0, 
                            silence_threshold=0.01)
        logger.info(f"User response: {user_response}")
        
        return user_response


async def serve() -> None:
    """Run the voice assistant MCP server using stdio transport."""
    logger.info("Starting Voice Assistant MCP server")
    
    server = Server("Voice Assistant")
    voice_server = VoiceAssistantServer()
    
    @server.list_tools()
    async def list_tools() -> list[Tool]:
        """List all available voice assistant tools."""
        return [
            Tool(
                name=VoiceTools.SPEAK.value,
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
                            "description": "The voice to use for speech. For Spanish, use one of 'ef_dora', 'em_alex', or 'em_santa'.",
                            "default": "af_bella"
                        },
                        "language": {
                            "type": "string",
                            "description": "The language code: 'en' (English), 'es' (Spanish). Spanish voices are 'ef_dora', 'em_alex', and 'em_santa'.",
                            "default": "en",
                            "enum": ["en", "es", "en-gb"]
                        }
                    },
                    "required": ["text"]
                }
            ),
            Tool(
                name=VoiceTools.LISTEN.value,
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
                name=VoiceTools.CONVERSATION_TURN.value,
                description="Speak a message and listen for a response in one step. IMPORTANT: This tool MUST be used for any interactive dialogue where you need user feedback. It combines speak and listen operations to ensure proper conversation flow. Use this whenever you need to ask the user a question and capture their response.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "system_message": {
                            "type": "string",
                            "description": "The message to speak to the user. For best results, make your questions clear and specific."
                        },
                        "max_listen_duration": {
                            "type": "integer",
                            "description": "Maximum duration to listen for a response (seconds)",
                            "default": 30
                        },
                        "language": {
                            "type": "string",
                            "description": "The language code: 'en' (English), 'es' (Spanish). Spanish voices are 'ef_dora', 'em_alex', and 'em_santa'.",
                            "default": "en",
                            "enum": ["en", "es", "en-gb"]
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
            match name:
                case VoiceTools.SPEAK.value:
                    text = arguments.get("text")
                    voice = arguments.get("voice", "af_bella")
                    language = arguments.get("language", "en")
                    
                    if not text:
                        raise ValueError("Missing required parameter: text")
                    
                    result = await voice_server.speak(text, voice, language)
                    return [TextContent(type="text", text=result)]
                
                case VoiceTools.LISTEN.value:
                    max_duration = arguments.get("max_duration", 30)
                    silence_duration = arguments.get("silence_duration", 2.0)
                    silence_threshold = arguments.get("silence_threshold", 0.01)
                    
                    result = await voice_server.listen(max_duration, silence_duration, silence_threshold)
                    return [TextContent(type="text", text=result)]
                
                case VoiceTools.CONVERSATION_TURN.value:
                    system_message = arguments.get("system_message")
                    max_listen_duration = arguments.get("max_listen_duration", 30)
                    language = arguments.get("language", "en")
                    
                    if not system_message:
                        raise ValueError("Missing required parameter: system_message")
                    
                    result = await voice_server.conversation_turn(system_message, max_listen_duration, language)
                    return [TextContent(type="text", text=result)]
                
                case _:
                    raise ValueError(f"Unknown tool: {name}")
        except Exception as e:
            logger.error(f"Error in call_tool: {str(e)}", exc_info=True)
            return [TextContent(type="text", text=f"Error: {str(e)}")]

    # Create server options
    options = server.create_initialization_options()
    
    # Using the stdio_server context manager for proper stdio handling
    async with stdio_server() as (read_stream, write_stream):
        logger.info("Voice Assistant MCP Server connected to stdio streams")
        try:
            # Run the server with explicit streams
            await server.run(read_stream, write_stream, options, raise_exceptions=False)
        except Exception as e:
            logger.error(f"Error running MCP server: {e}", exc_info=True)

if __name__ == "__main__":
    try:
        # Run the server using asyncio
        asyncio.run(serve())
    except KeyboardInterrupt:
        logger.info("Server stopped by KeyboardInterrupt (Ctrl+C).")
    except Exception as e:
        logger.critical(f"FATAL ERROR in MCP server: {e}", exc_info=True)
    finally:
        cleanup_resources()
        logger.info("MCP server process ending.") 