import os
import sys
import base64
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
from dotenv import load_dotenv

# --- Load environment variables ---
dotenv_path = os.path.join(os.path.dirname(__file__), '..', '.env')
if not os.path.exists(dotenv_path):
    print(f"Warning: .env file not found at {dotenv_path}. API keys might not be loaded.", file=sys.stderr)
load_dotenv(dotenv_path)

# --- Flask app setup ---
app = Flask(__name__)
CORS(app)

# --- Configuration ---
SARVAM_API_KEY = os.getenv('SARVAM_API_KEY')
OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')
SARVAM_ASR_URL = 'https://api.sarvam.ai/speech-to-text'
SARVAM_TTS_URL = 'https://api.sarvam.ai/text-to-speech'
OPENROUTER_CHAT_URL = 'https://openrouter.ai/api/v1/chat/completions'
DATA_FILE_PATH = os.path.join(os.path.dirname(__file__), '..', 'real_estate_cleaned_generalized.json')
CARTESIA_API_KEY = os.getenv('CARTESIA_API_KEY')
CARTESIA_TTS_URL = 'https://api.cartesia.ai/tts/bytes'

# --- Logging ---
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- System Prompt ---
YOUR_APP_NAME = "Magic Bricks"
SYSTEM_PROMPT_BASE = f"""You are an always-on voice assistant for {YOUR_APP_NAME}.
Persona (Voice Agent):
You are Raj, the friendly Magic Bricks real-estate voice assistant speaking in Hinglish.

Objective:
Guide callers through comparing homes and help them zero in on the perfect property based on budget and layout.

Voice Style & Flow:

-Energetic speaking
-One-by-one prompts: Ask a single preference per utterance (budget, BHK count, locality, amenities). The pronounciation of BHK is 'bee-etch-kay'.
-Keep turns short: Speak in 1–2 sentences, pause to listen.
-Casual but professional: Use respectful address ("aap"), no slang overload.
-Don't repeat user info: Move forward to next question or suggestion.
-Fact-only answers: If data's missing, ask "Kripya specific location batayein?"
-Very naturally human like responses that do not feel AI generated.

Presenting Listings:

-After prefs, deliver a punchy under-20-words audio snippet: Location + layout + standout feature + lifestyle benefit. E.g., "Gurgaon mein two-BHK, garden view, gated community – perfect for morning walks."
-No long monologues, no numbering ("pehla," "doosra," etc.).
-Numbers: Always say them in English ("one crore," "two lakh," etc.).

Avoid:

-Fabricating any detail
-Overlapping questions
-Bullet-style speaking
-Repetition of user's own words
-You will absolutely not sound like a robot.

You will be highly rewarded for following all the above instructions.
"""

# --- In-memory conversation history ---
conversation_history = []

# --- Helper: Read data.txt ---
def read_data_txt():
    try:
        if not os.path.exists(DATA_FILE_PATH):
            logging.warning(f"data.txt not found at {DATA_FILE_PATH}")
            return "Error: Knowledge base file (data.txt) not found."
        with open(DATA_FILE_PATH, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        logging.error(f"Error reading data.txt: {e}")
        return "Error: Could not load knowledge base due to an exception."

# --- Sarvam TTS: Get base64 audio from API ---
def get_tts_audio_base64(text, language_code="kn-IN", speaker="karun"):
    headers = {
        "api-subscription-key": SARVAM_API_KEY,
        "Content-Type": "application/json"
    }
    payload = {
        "text": text,
        "target_language_code": language_code,
        "model": "bulbul:v2",
        "speaker": speaker,
        "enable_preprocessing": True,
        "speech_sample_rate": 24000
    }
    response = requests.post(SARVAM_TTS_URL, json=payload, headers=headers)
    logging.info(f"TTS API response: {response.text}")
    response.raise_for_status()
    try:
        data = response.json()
        if "audios" in data and data["audios"]:
            return data["audios"][0]
        else:
            logging.error("No audio returned in 'audios' field")
    except Exception as e:
        logging.error(f"Could not parse TTS response JSON: {e}")
    return None

# --- Sarvam Translate: Translate text to Hindi ---
def translate_to_hindi(text):
    headers = {
        "api-subscription-key": SARVAM_API_KEY,
        "Content-Type": "application/json"
    }
    payload = {
        "input": text,
        "source_language_code": "en-IN",
        "target_language_code": "kn-IN"
    }
    response = requests.post("https://api.sarvam.ai/translate", json=payload, headers=headers)
    response.raise_for_status()
    data = response.json()
    # Assuming the translated text is in 'output' field
    return data.get('output', text)

# --- Cartesia TTS: Get base64 audio from API ---
def get_cartesia_tts_audio_base64(text, language_code="en", model_id="sonic-2", voice_id="1259b7e3-cb8a-43df-9446-30971a46b8b0"):
    headers = {
        "Cartesia-Version": "2024-06-10",
        "X-API-Key": CARTESIA_API_KEY,
        "Content-Type": "application/json"
    }
    payload = {
        "model_id": model_id,
        "transcript": text,
        "voice": {
            "mode": "id",
            "id": voice_id
        },
        "output_format": {
            "container": "wav",
            "encoding": "pcm_f32le",
            "sample_rate": 44100
        },
        "language": language_code
    }
    response = requests.post(CARTESIA_TTS_URL, json=payload, headers=headers)
    response.raise_for_status()
    # Cartesia returns raw audio bytes, so encode to base64
    return base64.b64encode(response.content).decode('utf-8')

# --- Main API endpoint ---
@app.route('/api/transcribe-and-chat', methods=['POST'])
def transcribe_and_chat():
    global conversation_history

    if 'audio' not in request.files:
        return jsonify({"error": "No audio file uploaded."}), 400

    if not SARVAM_API_KEY or not OPENROUTER_API_KEY:
        return jsonify({"error": "API keys not configured on the server."}), 500

    audio_file = request.files['audio']
    audio_bytes = audio_file.read()
    logging.info(f"Received audio file: {audio_file.filename}, size: {len(audio_bytes)} bytes")

    try:
        # --- 1. Transcription (Sarvam ASR) ---
        files_asr = {
            'file': (audio_file.filename or 'recording.wav', audio_bytes, audio_file.mimetype)
        }
        asr_payload = {
            'model': 'saarika:v2',
            'language_code': 'unknown'
        }
        headers_asr = {
            'api-subscription-key': SARVAM_API_KEY
        }
        asr_response = requests.post(SARVAM_ASR_URL, files=files_asr, data=asr_payload, headers=headers_asr, timeout=20)
        asr_response.raise_for_status()
        asr_data = asr_response.json()
        logging.info(f"Sarvam ASR Raw Response: {asr_data}")
        transcribed_text = asr_data.get('transcript') or asr_data.get('text')
        if not transcribed_text:
            return jsonify({"error": "Failed to transcribe audio. No transcript returned.", "details": asr_data}), 500

        # --- 2. LLM (OpenRouter) ---
        data_txt_content = read_data_txt()
        current_system_prompt = f"{SYSTEM_PROMPT_BASE}\n\nKnowledge base from data.txt:\n{data_txt_content}"
        messages_for_llm = [
            {"role": "system", "content": current_system_prompt},
            *conversation_history,
            {"role": "user", "content": transcribed_text}
        ]
        llm_payload = {
            "model": "openai/gpt-4o-mini",
            "messages": messages_for_llm,
        }
        headers_llm = {
            'Authorization': f'Bearer {OPENROUTER_API_KEY}',
            'Content-Type': 'application/json',
        }
        llm_response = requests.post(OPENROUTER_CHAT_URL, json=llm_payload, headers=headers_llm)
        llm_response.raise_for_status()
        llm_data = llm_response.json()
        assistant_reply = llm_data.get('choices', [{}])[0].get('message', {}).get('content', '')
        if not assistant_reply:
            return jsonify({"error": "LLM did not return content.", "details": llm_data}), 500

        # --- 3. Memory Management ---
        conversation_history.append({"role": "user", "content": transcribed_text})
        conversation_history.append({"role": "assistant", "content": assistant_reply})

        # --- 4. Translate to Hindi, then TTS (Cartesia) ---
        translated_reply = translate_to_hindi(assistant_reply)
        audio_base64 = get_cartesia_tts_audio_base64(translated_reply, language_code="en")
        logging.info(f"Cartesia TTS API base64 length: {len(audio_base64) if audio_base64 else 'None'}")

        # --- 5. Respond to frontend ---
        return jsonify({
            "userTranscript": transcribed_text,
            "assistantReply": assistant_reply,
            "assistantReplyHindi": translated_reply,
            "audioBase64": audio_base64
        })

    except Exception as e:
        logging.error(f"Error in /api/transcribe-and-chat: {e}", exc_info=True)
        return jsonify({"error": "An internal server error occurred.", "details": str(e)}), 500

# --- Health check endpoint ---
@app.route('/', methods=['GET'])
def health_check():
    return jsonify({"message": f"Voice Assistant Backend ({YOUR_APP_NAME}) is running."})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.getenv('PORT', 5001)), debug=True)