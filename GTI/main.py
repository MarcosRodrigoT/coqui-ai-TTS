import torch
from TTS.api import TTS


# Parameters for speech generation
USER = "mrt"
USER_VOICE = f"./GTI/users/{USER}.wav"
OUTPUT_PATH = f"./GTI/outputs/{USER}.wav"
text = "Hola, somos el grupo de tratamiento de imágenes. Nos dedicamos a muchos temas distintos, como son las redes neuronales y la visión artificial."

# Get device
device = "cuda" if torch.cuda.is_available() else "cpu"

# # List available 🐸TTS models
# print("List of available 🐸TTS models:")
# print(TTS().list_models())

# Load model
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

# # List speakers
# print("List of available speakers:")
# print(tts.speakers)

# Generate speech by cloning a voice using default settings
tts.tts_to_file(
    text=text,
    speaker_wav=USER_VOICE,
    # speaker="Viktor Menelaos",
    language="es",
    file_path=OUTPUT_PATH,
)
