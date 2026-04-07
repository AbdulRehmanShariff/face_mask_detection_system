from gtts import gTTS
import os

alerts = [
    ("mask_laga.mp3",  "Bhai, mask lagao! Kya yaar!"),
    ("mask_kahan.mp3", "Arre bhai! Mask kahan hai tera?"),
    ("mask_wear.mp3",  "Oye! Mask pehno jaldi se!"),
]

for filename, text in alerts:
    print(f"🎙️  Generating: {filename} → '{text}'")
    tts = gTTS(text=text, lang="hi", slow=False)
    tts.save(filename)
    print(f" Saved: {filename}")

print("\n All audio files generated!")
print("Now run webcam_hindi_alert.py to use them!")