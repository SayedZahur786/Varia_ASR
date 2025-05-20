from faster_whisper import WhisperModel

print("model started")

model_size = "small.en"
#Using Small.en for cpu when we will deploy we will use larger-v3


model = WhisperModel(model_size, device="cuda", compute_type="float16")

#We will use cuda for gpu and fp32 at scaling


segments, info = model.transcribe("audio.mp3",language="en" ,beam_size=5)

# print("Detected language '%s' with probability %f" % (info.language, info.language_probability))

for segment in segments:
    print(segment.text)