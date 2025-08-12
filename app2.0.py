<<<<<<< HEAD
from langchain_ollama import OllamaLLM
from configlama import OLLAMA_API_URL
import sounddevice as sd
import numpy as np
import whisper
import torch
import simpleaudio as sa
from pydub import AudioSegment
import os
import tempfile
import time
import warnings
from TTS.api import TTS
from silero_vad import VADIterator, get_speech_timestamps, collect_chunks

# Load TTS model
tts = TTS(model_name="tts_models/en/ljspeech/vits--neon", progress_bar=False).to("cpu")

# Load Silero VAD
model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', trust_repo=True)
(get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = utils
vad = VADIterator(model)

# Temp folder
TEMP_DIR = "D:/VisualSC/MyChatbotRAG/TempPython"
os.makedirs(TEMP_DIR, exist_ok=True)
warnings.filterwarnings("ignore", category=UserWarning)

# LLM Model (Ollama)
llm = OllamaLLM(
    model="llama2:7b-chat",
    base_url="http://localhost:11434"
)

def pitcher(input_file, pitch=0.2):
    sound = AudioSegment.from_file(input_file, format="wav")
    new_sample_rate = int(sound.frame_rate * (2.0 ** pitch))
    high_pitch_sound = sound._spawn(sound.raw_data, overrides={'frame_rate' : new_sample_rate})
    high_pitch_sound = high_pitch_sound.set_frame_rate(22050)
    return np.array(high_pitch_sound.get_array_of_samples()).astype(np.float32) / 32768  # convert ke float32 numpy

def rekam_audio(durasi=5, fs=16000):
    print("🎙️ Mulai merekam...")
    audio = sd.rec(int(durasi * fs), samplerate=fs, channels=1, dtype='float32')
    sd.wait()
    print("✅ Rekaman selesai.")
    return np.squeeze(audio), fs

def simpan_wav(nama_file, audio, fs):
    import soundfile as sf
    sf.write(nama_file, audio, fs)
    print(f"💾 Audio disimpan di {nama_file}")
    time.sleep(0.5)

def stt_whisper(audio_file):
    model = whisper.load_model("small")
    result = model.transcribe(audio_file, language='en')
    return result["text"]

def chatAI(user_input):
    result = llm.invoke(user_input)
    return result

def tts_coqui_simpleaudio(teks):
    # Buat file sementara dari TTS
    temp_wav = os.path.join(TEMP_DIR, "tts_temp.wav")
    tts.tts_to_file(text=teks, file_path=temp_wav)

    # Pitch shift
    audio_data = pitcher(temp_wav, pitch=0.2)

    # Convert float32 numpy ke int16 PCM
    audio_int16 = (audio_data * 32767).astype(np.int16)

    # Mainkan dengan simpleaudio
    play_obj = sa.play_buffer(audio_int16, 1, 2, 22050)
    play_obj.wait_done()

    # Bersihkan file sementara
    try:
        os.remove(temp_wav)
    except:
        pass

# MAIN LOOP
exit_commands = ["stop", "quit", "exit", "end conversation"]

while True:
    durasi_rekam = 3  # Detik
    audio, fs = rekam_audio(durasi=durasi_rekam)
    speech_timestamps = get_speech_timestamps(audio, model, sampling_rate=fs)

    if len(speech_timestamps) == 0:
        print("❌ Tidak ada suara, skip turn.")
        continue

    # Simpan rekaman
    file_wav = os.path.join(TEMP_DIR, "rekaman.wav")
    simpan_wav(file_wav, audio, fs)

    teks_input = stt_whisper(file_wav)
    if teks_input.strip() == "":
        print("❌ Tidak ada teks yang dikenali dari suara, skip turn.")
        continue

    print(f"📝 Teks dari suara: {teks_input}")
    if any(cmd in teks_input.lower() for cmd in exit_commands):
        print("\n🚪 Perintah keluar terdeteksi. Menghentikan chatbot...")
        break

    jawaban_ai = chatAI(teks_input)
    print(f"🤖 Jawaban AI: {jawaban_ai}")

    tts_coqui_simpleaudio(jawaban_ai)
=======
from langchain_ollama import OllamaLLM
from configlama import OLLAMA_API_URL
import sounddevice as sd
import numpy as np
import whisper
import torch
import simpleaudio as sa
from pydub import AudioSegment
import os
import tempfile
import time
import warnings
from TTS.api import TTS
from silero_vad import VADIterator, get_speech_timestamps, collect_chunks

# Load TTS model
tts = TTS(model_name="tts_models/en/ljspeech/vits--neon", progress_bar=False).to("cpu")

# Load Silero VAD
model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', trust_repo=True)
(get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = utils
vad = VADIterator(model)

# Temp folder
TEMP_DIR = "D:/VisualSC/MyChatbotRAG/TempPython"
os.makedirs(TEMP_DIR, exist_ok=True)
warnings.filterwarnings("ignore", category=UserWarning)

# LLM Model (Ollama)
llm = OllamaLLM(
    model="llama2:7b-chat",
    base_url="http://localhost:11434"
)

def pitcher(input_file, pitch=0.2):
    sound = AudioSegment.from_file(input_file, format="wav")
    new_sample_rate = int(sound.frame_rate * (2.0 ** pitch))
    high_pitch_sound = sound._spawn(sound.raw_data, overrides={'frame_rate' : new_sample_rate})
    high_pitch_sound = high_pitch_sound.set_frame_rate(22050)
    return np.array(high_pitch_sound.get_array_of_samples()).astype(np.float32) / 32768  # convert ke float32 numpy

def rekam_audio(durasi=5, fs=16000):
    print("🎙️ Mulai merekam...")
    audio = sd.rec(int(durasi * fs), samplerate=fs, channels=1, dtype='float32')
    sd.wait()
    print("✅ Rekaman selesai.")
    return np.squeeze(audio), fs

def simpan_wav(nama_file, audio, fs):
    import soundfile as sf
    sf.write(nama_file, audio, fs)
    print(f"💾 Audio disimpan di {nama_file}")
    time.sleep(0.5)

def stt_whisper(audio_file):
    model = whisper.load_model("small")
    result = model.transcribe(audio_file, language='en')
    return result["text"]

def chatAI(user_input):
    result = llm.invoke(user_input)
    return result

def tts_coqui_simpleaudio(teks):
    # Buat file sementara dari TTS
    temp_wav = os.path.join(TEMP_DIR, "tts_temp.wav")
    tts.tts_to_file(text=teks, file_path=temp_wav)

    # Pitch shift
    audio_data = pitcher(temp_wav, pitch=0.2)

    # Convert float32 numpy ke int16 PCM
    audio_int16 = (audio_data * 32767).astype(np.int16)

    # Mainkan dengan simpleaudio
    play_obj = sa.play_buffer(audio_int16, 1, 2, 22050)
    play_obj.wait_done()

    # Bersihkan file sementara
    try:
        os.remove(temp_wav)
    except:
        pass

# MAIN LOOP
exit_commands = ["stop", "quit", "exit", "end conversation"]

while True:
    durasi_rekam = 3  # Detik
    audio, fs = rekam_audio(durasi=durasi_rekam)
    speech_timestamps = get_speech_timestamps(audio, model, sampling_rate=fs)

    if len(speech_timestamps) == 0:
        print("❌ Tidak ada suara, skip turn.")
        continue

    # Simpan rekaman
    file_wav = os.path.join(TEMP_DIR, "rekaman.wav")
    simpan_wav(file_wav, audio, fs)

    teks_input = stt_whisper(file_wav)
    if teks_input.strip() == "":
        print("❌ Tidak ada teks yang dikenali dari suara, skip turn.")
        continue

    print(f"📝 Teks dari suara: {teks_input}")
    if any(cmd in teks_input.lower() for cmd in exit_commands):
        print("\n🚪 Perintah keluar terdeteksi. Menghentikan chatbot...")
        break

    jawaban_ai = chatAI(teks_input)
    print(f"🤖 Jawaban AI: {jawaban_ai}")

    tts_coqui_simpleaudio(jawaban_ai)
>>>>>>> 46f8efefacdbcef7e0ed266c41059ff89321cbc6
