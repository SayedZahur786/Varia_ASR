import sounddevice as sd
import numpy as np
import queue
import threading
import scipy.signal
from faster_whisper import WhisperModel

# Custom Setting for Live Audio
samplerate = 16000
block_duration = 0.5         # Keep it short for responsiveness
chunk_duration = 1.5         # Increased chunk for better context
channels = 1

frames_per_block = int(samplerate * block_duration)
frames_per_chunk = int(samplerate * chunk_duration)

audio_queue = queue.Queue()
audio_buffer = []

# ✅ Use Medium model for higher accuracy
model = WhisperModel("medium.en", device="cuda", compute_type="float16")

# ✅ Optional: Denoise audio using Butterworth filter
def denoise_audio(audio):
    b, a = scipy.signal.butter(6, 0.05)
    return scipy.signal.filtfilt(b, a, audio)

def audio_callback(indata, frames, time, status):
    if status:
        print(status)
    audio_queue.put(indata.copy())

def recorder():
    with sd.InputStream(samplerate=samplerate, channels=channels,
                        dtype='float32',
                        callback=audio_callback, blocksize=frames_per_block):
        print("🎤 Listening... press Ctrl+C to stop")
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

            # ✅ Optional denoising (you can comment this if not needed)
            audio_data = denoise_audio(audio_data)

            # ✅ GPU Transcription with better accuracy
            segments, _ = model.transcribe(
                audio_data,
                language="en",
                beam_size=5,
                # Helps reduce stutters/silence
            )

            for segment in segments:
                print("🗣️", segment.text)

# 🎬 Start recording in a separate thread
threading.Thread(target=recorder, daemon=True).start()
transcriber()
