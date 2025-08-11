from langchain_ollama import OllamaLLM
from configlama import OLLAMA_API_URL
import sounddevice as sd
import numpy as np
import winsound
import simpleaudio as sa
import whisper
import torch
from gtts import gTTS
from pydub import AudioSegment
from pydub.playback import play
import os
import tempfile
import time
import warnings
from TTS.api import TTS
from silero_vad import VADIterator, get_speech_timestamps, collect_chunks

tts = TTS(model_name="tts_models/en/ljspeech/vits--neon").to("cpu")

model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', trust_repo=True)
(get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = utils
vad = VADIterator(model)

tempfile.tempdir = "D:/VisualSC/MyChatbotRAG/TempPython"
os.makedirs(tempfile.tempdir, exist_ok=True)

warnings.filterwarnings("ignore", category=UserWarning)

# LLM Model
llm = OllamaLLM(
    model="llama2:7b-chat",
    base_url="http://localhost:11434"
)

def pitcher(input_file, output_file, pitch=0.2):
    sound = AudioSegment.from_file(input_file, format="wav")
    new_sample_rate = int(sound.frame_rate * (2.0 ** pitch))
    high_pitch_sound = sound._spawn(sound.raw_data, overrides={'frame_rate' : new_sample_rate})
    high_pitch_sound = high_pitch_sound.set_frame_rate(22050)
    high_pitch_sound.export(output_file, format="wav")
    
def rekam_audio(durasi=5, fs=16000):
    print("Mulai merekam...")
    audio = sd.rec(int(durasi * fs), samplerate=fs, channels=1, dtype='float32')
    sd.wait()
    print("Rekaman selesai.")
    audio = np.squeeze(audio)
    return audio, fs

def simpan_wav(nama_file, audio, fs):
    import soundfile as sf
    sf.write(nama_file, audio, fs)
    print(f"Audio disimpan di {nama_file}")
    time.sleep(0.5)

def stt_whisper(audio_file):
    model = whisper.load_model("small")
    result = model.transcribe(audio_file, language='en')
    return result["text"]

def chatAI(user_input):
    result = llm.invoke(user_input)
    return result

def tts_coqui_playback(teks):
    temp_wav = os.path.join(tempfile.tempdir, "tts_temp.wav")
    temp_pitch_wav = os.path.join(tempfile.tempdir, "tts_pitch_temp.wav")
    
    tts.tts_to_file(text=teks, file_path=temp_wav)
    
    pitcher(temp_wav, temp_pitch_wav, pitch=0.2)
    
    sound = AudioSegment.from_file(temp_pitch_wav, format="wav")

    winsound.PlaySound(temp_pitch_wav, winsound.SND_FILENAME)
    
###    winsound.PlaySound(temp_wav, winsound.SND_FILENAME)
    try:
        os.remove(temp_wav)
        os.remove(temp_pitch_wav)
    except:
        pass


# MAIN
exit_commands = ["stop", "quit", "exit", "end conversation"]

while True:
    durasi_rekam = 6  # Detik
    audio, fs = rekam_audio(durasi=durasi_rekam)
    speech_timestamps = get_speech_timestamps(audio, model, sampling_rate=fs)
    if len(speech_timestamps) == 0:
        print("❌ Tidak ada suara, skip turn.")
        continue

    # Simpan file rekaman ke D:
    file_wav = os.path.join(tempfile.tempdir, "rekaman.wav")
    simpan_wav(file_wav, audio, fs)

    teks_input = stt_whisper(file_wav)
    
    if teks_input.strip() == "":
        print("❌ Tidak ada teks yang dikenali dari suara, skip turn.")
        continue
    
    print(f"Teks dari suara: {teks_input}")
    if any(cmd in teks_input.lower() for cmd in exit_commands):
        print("\n🚪 Perintah keluar terdeteksi. Menghentikan chatbot...")
        break

    jawaban_ai = chatAI(teks_input)
    print(f"Jawaban AI: {jawaban_ai}")

    tts_coqui_playback(jawaban_ai)
