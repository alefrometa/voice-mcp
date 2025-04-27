# Voice Assistant MCP Server

This project implements a Multi-Capability Protocol (MCP) server that provides voice assistant functionalities, including Text-to-Speech (TTS) and Speech-to-Text (STT) capabilities.

## Features

- **Text-to-Speech (TTS):** Converts text into audible speech using the Kokoro TTS engine.
- **Speech-to-Text (STT):** Transcribes spoken audio into text using the OpenAI Whisper model.
- **Conversation Turn:** Combines TTS and STT into a single operation for seamless conversation flow.
- **Silence Detection:** Automatically stops recording audio when silence is detected, with configurable thresholds and durations.
- **Background Noise Calibration:** Adjusts the silence detection threshold based on ambient noise levels.

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd voice-mcp
    ```
2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```
3.  **Install dependencies:**
    Make sure you have `portaudio` installed (`brew install portaudio` on macOS, `sudo apt-get install portaudio19-dev python3-pyaudio` on Debian/Ubuntu). Then install Python packages:
    ```bash
    pip install -r requirements.txt
    ```
    _(Note: A `requirements.txt` file should be created containing necessary packages like `openai-whisper`, `torch`, `pyaudio`, `kokoro-tts`, etc.)_

## Usage

The server is designed to be run as an MCP process, typically integrated with a client application like Cursor or Anysphere.

1.  **Start the server:** The client application will usually manage starting the server process based on its configuration.
2.  **Interact via MCP:** The client application can then call the server's tools (`speak`, `listen`, `conversation_turn`) using the MCP protocol.

## Configuration (Example for MCP Client)

To use this server with an MCP client (like Cursor/Anysphere), you need to configure the client to run the `voice_server.py` script. Here's a generic example of how such a configuration might look in a JSON file:

```json
{
  "Voice Assistant Server": {
    // The name you give the server in the client
    "command": "/path/to/your/project/voice-mcp/venv/bin/python", // Absolute path to python in venv
    "args": [
      "-u", // Unbuffered output
      "/path/to/your/project/voice-mcp/voice_server.py" // Absolute path to the server script
    ],
    "cwd": "/path/to/your/project/voice-mcp", // Absolute path to the project directory
    "timeout": 300 // Optional timeout in seconds
  }
}
```

**Important:** Replace `/path/to/your/project/voice-mcp` with the actual absolute path to the cloned repository on your system.

## Dependencies

- Python 3.9+
- PyAudio (requires PortAudio system library)
- OpenAI Whisper
- Kokoro TTS
- PyTorch
- NumPy
- (Potentially others - generate a `requirements.txt` for a full list)

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues.

## License

(Specify your license here, e.g., MIT License)
