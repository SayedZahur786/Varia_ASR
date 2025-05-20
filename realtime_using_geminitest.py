import sounddevice as sd
import numpy as np
import queue
import threading
from faster_whisper import WhisperModel
import google.generativeai as genai

# Gemini setup
genai.configure(api_key="AIzaSyB-bLPUaEtXYIFWYwc1UlhQ5nsitOz8JRc")
gemini_model = genai.GenerativeModel("models/gemini-pro")

def analyze_sentiment_with_gemini(text):
    prompt = f"Analyze the sentiment of the following sentence and label it as Positive, Negative, or Neutral. Also give a confidence score out of 100.\n\nSentence: \"{text}\""
    response = gemini_model.generate_content(prompt)
    return response.text

# Audio Settings
samplerate = 16000
block_duration = 1.5
chunk_duration = 1
channels = 1

frames_per_block = int(samplerate * block_duration)
frames_per_chunk = int(samplerate * chunk_duration)

audio_queue = queue.Queue()
audio_buffer = []

# Whisper Model on GPU
whisper_model = WhisperModel("medium.en", device="cuda", compute_type="float16")  # ✅ Renamed

def audio_callback(indata, frames, time, status):
    if status:
        print(status)
    audio_queue.put(indata.copy())

def recorder():
    with sd.InputStream(samplerate=samplerate, channels=channels,
                        dtype='float32',
                        callback=audio_callback, blocksize=frames_per_block):
        print("Listening .. press ctrl+c to stop")
        while True:
            sd.sleep(100)

def transcriber():
    global audio_buffer
    while True:
        block = audio_queue.get()
        audio_buffer.append(block)

        total_frames = sum(len(b) for b in audio_buffer)
        if total_frames >= frames_per_chunk:
            audio_data = np.concatenate(audio_buffer)[:frames_per_chunk]
            audio_buffer = []
            audio_data = audio_data.flatten().astype(np.float32)

            segments, _ = whisper_model.transcribe(
                audio_data,
                language="en",
                beam_size=1,
                vad_filter=True
            )

            for segment in segments:
                text = segment.text
                print(f"User: {text}")

                sentiment = analyze_sentiment_with_gemini(text)
                print(f"Gemini Sentiment Analysis: {sentiment}")

threading.Thread(target=recorder, daemon=True).start()
transcriber()
