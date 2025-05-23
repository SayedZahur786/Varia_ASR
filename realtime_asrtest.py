import sounddevice as sd
import numpy as np
import queue
import threading
from faster_whisper import WhisperModel
from transformers import pipeline

from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

model_name = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

sentiment_pipeline = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer, device=0)

# Custom Setting for Live Audio (DONE BY SAYED)
samplerate = 16000
block_duration = 0.5
chunk_duration = 1.5
channels = 1

frames_per_block = int(samplerate * block_duration)
frames_per_chunk = int(samplerate * chunk_duration)

audio_queue = queue.Queue()
audio_buffer = []

# ✅ Load GPU-optimized model
model = WhisperModel("medium.en", device="cuda", compute_type="float16")

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

            # ✅ Fast GPU transcription
            segments, _ = model.transcribe(
                audio_data,
                language="en",
                beam_size=1,
                vad_filter=True  # Optional: helps reduce silence/stutters
            )

            for segment in segments:
                text = segment.text
                print(f"User: {text}")
                sentiment = sentiment_pipeline(text)[0]
                print(f"Sentiment: {sentiment['label']} (score: {sentiment['score']:.2f})")

threading.Thread(target=recorder, daemon=True).start()
transcriber()
