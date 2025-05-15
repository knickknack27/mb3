# Real-Time RAG-Enabled Voice Assistant

This project implements a real-time voice assistant using Sarvam ASR for transcription and an OpenRouter-backed LLM with Retrieval Augmented Generation (RAG) capabilities.

## Features

*   Real-time voice transcription.
*   RAG using a `data.txt` file as the knowledge base.
*   Full conversational memory.
*   Secure API key management via environment variables.
*   Backend handles ASR and LLM calls.
*   Simple frontend for audio recording and displaying responses.

## Project Structure

```
/
├── backend/
│   ├── node_modules/
│   ├── server.js           # Express server
│   └── package.json
│   └── package-lock.json
├── frontend/
│   ├── index.html          # Main HTML file
│   ├── style.css           # CSS styles
│   └── script.js           # Frontend JavaScript
├── .env                    # Environment variables (ignored by git)
├── .env.example            # Example environment variables
├── data.txt                # Knowledge base for RAG
└── README.md
```

## Setup

1.  **Clone the repository (if applicable).**
2.  **Create a Python Virtual Environment (Recommended):**
    ```bash
    # In the project root
    python -m venv venv
    # Activate it (Windows)
    .\venv\Scripts\activate
    # Or (macOS/Linux)
    # source venv/bin/activate
    ```
3.  **Install Backend Dependencies:**
    ```bash
    # Ensure your virtual environment is active
    pip install -r backend/requirements.txt
    ```
4.  **Create Environment File:**
    *   Duplicate `dotenv.example` (or `.env.example` if you renamed it) and rename it to `.env` in the project root.
    *   Open `.env` and add your API keys:
        ```env
        SARVAM_API_KEY=your_sarvam_api_key_here
        OPENROUTER_API_KEY=your_openrouter_api_key_here
        # FLASK_DEBUG=True # Optional: for development mode
        # PORT=5001 # Optional: if you want to change the default port
        ```
5.  **Populate `data.txt`:**
    *   Add the content you want the assistant to use as its knowledge base into `data.txt` in the project root.

## Running the Assistant

1.  **Start the Backend Server:**
    ```bash
    # Ensure your virtual environment is active (if you created one)
    # From the project root directory:
    python backend/app.py
    ```
    The server will typically start on `http://localhost:5001` (or the port you configured).

2.  **Open the Frontend:**
    *   Open `frontend/index.html` in your web browser.

## How It Works

1.  The user clicks the "mic" button on the frontend.
2.  The browser records audio until silence or the user clicks "stop."
3.  The recorded audio is sent to the backend.
4.  The backend:
    *   Transcribes the audio using Sarvam's ASR API.
    *   Constructs a message history including the system prompt, `data.txt` content, past conversation, and the new transcript.
    *   Sends this to an OpenRouter LLM.
    *   Receives the LLM's response.
    *   Sends the response back to the frontend.
5.  The frontend displays the assistant's response.
6.  The loop continues if the mic is still "on." 