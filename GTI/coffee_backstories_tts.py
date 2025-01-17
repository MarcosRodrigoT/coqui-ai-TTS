import os
import torch
from TTS.api import TTS


# Coffee backstories parameters
BACKSTORIES_DIR = "/home/mrt/Projects/pix2pix/backstories"
VOICES_DIR = f"/home/mrt/Projects/coqui-ai-TTS/GTI/voices"

# Get device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load TTS model
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

# Convert each user's backstory to speech
for user in sorted(os.listdir(BACKSTORIES_DIR)):
    print(f"Converting {user} backstory to speech")

    # Load user's backstory
    with open(os.path.join(BACKSTORIES_DIR, user, "backstory.txt"), "r") as f:
        backstory = f.read()

    # Load user's voice if exists, if not use default voice
    voice = os.path.join(VOICES_DIR, f"{user}.wav")
    if os.path.exists(voice):
        tts.tts_to_file(
            text=backstory,
            # TODO: Change at some point to use user's voice
            # speaker_wav=voice,
            speaker="Viktor Menelaos",
            language="en",
            file_path=os.path.join(BACKSTORIES_DIR, user, "speech.wav"),
        )
    else:
        tts.tts_to_file(
            text=backstory,
            speaker="Viktor Menelaos",
            language="en",
            file_path=os.path.join(BACKSTORIES_DIR, user, "speech.wav"),
        )
