from faster_whisper import WhisperModel

model = WhisperModel(
    "base",
    device="cpu",
    compute_type="int8"
)

segments, info = model.transcribe("speech.wav")

print("Language:", info.language)

for segment in segments:
    print(segment.text)
    