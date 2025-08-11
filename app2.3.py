from langchain_ollama import OllamaLLM
from configlama import OLLAMA_API_URL
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_community.document_loaders import UnstructuredWordDocumentLoader
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import sounddevice as sd
import re
import logging
import numpy as np
import whisper
import torch
import json
import simpleaudio as sa
from pydub import AudioSegment
import os
import tempfile
import time
import warnings
from TTS.api import TTS
from silero_vad import VADIterator, get_speech_timestamps, collect_chunks
import pickle

MTIME_RECORD = "D:/VisualSC/MyChatbotRAG/vector_store/.mtime_record.pkl"

logging.basicConfig(level=logging.INFO, filename='D:/VisualSC/MyChatbotRAG/chatbot_log.txt', filemode='a', format='%(asctime)s - %(levelname)s - %(message)s')

urls = ["https://en.wikipedia.org/wiki/AI", "https://en.wikipedia.org/wiki/NLP"]
loader = WebBaseLoader(
    web_path=urls
)
docs = loader.load()

# Penyimpanan Chat History
chat_history_dir = "D:/VisualSC/MyChatbotRAG/chat_history"
os.makedirs(chat_history_dir, exist_ok=True)
chat_history_file = os.path.join(chat_history_dir, "chat_log.json")

memory = ConversationBufferMemory(return_messages=True)

# Penyimpanan vector history (Database)
VECTOR_STORE_DIR = "D:/VisualSC/MyChatbotRAG/vector_store"
os.makedirs(VECTOR_STORE_DIR, exist_ok=True)

embedding_model = HuggingFaceEmbeddings(model_name = "BAAI/bge-small-en")

# Load TTS model
tts = TTS(model_name="tts_models/en/ljspeech/vits--neon", progress_bar=False).to("cuda")

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
    model="phi3:mini",
    base_url="http://localhost:11434",
    streaming = True
)

# Personality
rag_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a friendly person speaking in casual tone. You can feel happy, sad, and angry. Give short, direct answers. No roleplay, no emoji, no action tags. No long assistant disclaimers. You can use this our history chat here:\n\n{context}"),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{question}")
])


def build_vectorstore():
    docs = []
    for file in os.listdir("D:/VisualSC/MyChatbotRAG/data_source"):
        file_path = os.path.join("D:/VisualSC/MyChatbotRAG/data_source", file)
        if file.endswith(".txt"):
            loader = TextLoader(file_path, encoding="utf-8")
        elif file.endswith("pdf"):
            loader = PyPDFLoader(file_path)
        elif file.endswith(".docx"):
            loader = UnstructuredWordDocumentLoader(file_path)
        else:
            continue
        docs.extend(loader.load())
        
    splitter = RecursiveCharacterTextSplitter(chunk_size = 250, chunk_overlap = 50)
    texts = splitter.split_documents(docs)
    
    vectorstore = FAISS.from_documents(texts, embedding_model)
    vectorstore.save_local(VECTOR_STORE_DIR)
    print("Vectorestore built and saved")
    return vectorstore

def load_vectorstore():
    return FAISS.load_local(VECTOR_STORE_DIR, embedding_model, allow_dangerous_deserialization = True)

def save_chat_history(user_input, bot_response):
    history = []
    if os.path.exists(chat_history_file):
        with open(chat_history_file, "r", encoding="utf-8") as f:
            history = json.load(f)
    history.append({"user": user_input, "bot": bot_response})
    with open(chat_history_file, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
        
def load_chat_history_to_memory():
    if os.path.exists(chat_history_file):
        try:
            with open(chat_history_file, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if content:  # Cek kalau file tidak kosong
                    history = json.loads(content)
                    for entry in history:
                        memory.chat_memory.add_user_message(entry["user"])
                        memory.chat_memory.add_ai_message(entry["bot"])
                else:
                    print("📂 Chat history kosong, skip load.")
        except json.JSONDecodeError:
            print("⚠️ Chat history corrupt atau bukan JSON valid, skip load.")

def pitcher(input_file, pitch=0.3):
    sound = AudioSegment.from_file(input_file, format="wav")
    new_sample_rate = int(sound.frame_rate * (2.0 ** pitch))
    high_pitch_sound = sound._spawn(sound.raw_data, overrides={'frame_rate' : new_sample_rate})
    high_pitch_sound = high_pitch_sound.set_frame_rate(22050)
    return np.array(high_pitch_sound.get_array_of_samples()).astype(np.float32) / 32768  # convert ke float32 numpy

def rekam_dengan_postbuffer(durasi=4, post_buffer=1, fs=16000):
    total_durasi = durasi + post_buffer
    print(f"🎙️ Mulai merekam {total_durasi} detik (post-buffer {post_buffer} detik)...")
    audio = sd.rec(int(total_durasi * fs), samplerate=fs, channels=1, dtype='float32')
    sd.wait()
    print("✅ Rekaman selesai.")

    # Deteksi bagian yang ada suara
    speech_timestamps = get_speech_timestamps(np.squeeze(audio), model, sampling_rate=fs)
    if len(speech_timestamps) == 0:
        print("❌ Tidak ada suara yang terdeteksi.")
        return None, fs

    start = speech_timestamps[0]['start']
    end = speech_timestamps[-1]['end']

    # Tambahkan post-buffer (1 detik di belakang akhir suara)
    end_with_buffer = min(end + int(post_buffer * fs), len(audio))

    final_audio = audio[start:end_with_buffer]
    return np.squeeze(final_audio), fs


def simpan_wav(nama_file, audio, fs):
    import soundfile as sf
    sf.write(nama_file, audio, fs)
    print(f"💾 Audio disimpan di {nama_file}")
    time.sleep(0.5)

whisper_model = whisper.load_model("small").to("cpu")

def stt_whisper(audio_file):
    result = whisper_model.transcribe(audio_file, language='en')
    return result["text"]

def chatAI_stream_and_tts(user_input):
    buffer_text = ""
    full_response = ""
    
    def token_callback(token):
        nonlocal buffer_text, full_response
        buffer_text += token
        full_response += token

        # Kalau sudah sampai titik, tanya, atau seru → Langsung TTS
        if any(p in buffer_text for p in [".", "!", "?"]):
            print("\nCihaku:", buffer_text.strip())
            tts_coqui_simpleaudio(buffer_text.strip())
            buffer_text = ""

    # Stream dari LLM
    for chunk in rag_chain.stream({"input": user_input}):
        if hasattr(chunk, "content"):
            token = chunk.content
        else:
            token = str(chunk)
        token_callback(token)

    # Kalau masih ada sisa teks yang belum diputar
    if buffer_text.strip():
        print("\nCihaku:", buffer_text.strip())
        tts_coqui_simpleaudio(buffer_text.strip())
        
    return full_response.strip()

def speed_change(sound, speed=1.0):
    altered = sound._spawn(sound.raw_data, overrides={
        "frame_rate": int(sound.frame_rate * speed)
    })
    return altered.set_frame_rate(22050)

def apply_rvc(input_wav, output_wav):
    import subprocess
    subprocess.run([
        "python", "inference_main.py",
        "-m", "D:/VisualSC/MyChatbotRAG/rvc_models/Kanade1/model.pth",
        "-c", "D:/VisualSC/MyChatbotRAG/rvc_models/Kanade1/config.json",
        "-n", input_wav,
        "-o", output_wav,
        "--index_path", "D:/VisualSC/MyChatbotRAG/rvc_models/Kanade1/added_IVF.index",
        "--f0method", "crepe"
    ])


def tts_coqui_simpleaudio(teks, speed=0.9, pitch_shift=0.3):
    
    temp_wav = os.path.join(TEMP_DIR, "tts_temp.wav")
    tts.tts_to_file(text=teks, file_path=temp_wav)

    sound = AudioSegment.from_file(temp_wav, format="wav")

    sound = speed_change(sound, speed=speed)

    new_sample_rate = int(sound.frame_rate * (2.0 ** pitch_shift))
    sound = sound._spawn(sound.raw_data, overrides={'frame_rate': new_sample_rate})
    sound = sound.set_frame_rate(22050)
    
    audio_data = np.array(sound.get_array_of_samples()).astype(np.float32) / 32768

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

def filter_words(teks):
    bad_words = ["badword1", "badword2"]
    for word in bad_words:
        teks = teks.replace(word, "***")
    return teks

def get_file_mtimes(folder):
    return {
        fname: os.path.getmtime(os.path.join(folder, fname))
        for fname in os.listdir(folder)
        if fname.endswith((".txt", ".pdf", ".docx"))
    }

def auto_update_vectorstore_with_mtime():
    source_dir = "D:/VisualSC/MyChatbotRAG/data_source"
    index_path = os.path.join(VECTOR_STORE_DIR, "index")

    current_mtimes = get_file_mtimes(source_dir)

    # Belum ada vectorstore atau record waktu
    if not os.path.exists(index_path) or not os.path.exists(MTIME_RECORD):
        print("📂 Vectorstore belum ada atau tidak ada record waktu, membangun ulang...")
        vectorstore = build_vectorstore()
        with open(MTIME_RECORD, "wb") as f:
            pickle.dump(current_mtimes, f)
        return vectorstore

    # Bandingkan waktu modifikasi terakhir
    with open(MTIME_RECORD, "rb") as f:
        last_mtimes = pickle.load(f)

    if current_mtimes != last_mtimes:
        print("✏️ Deteksi perubahan isi file. Rebuilding vectorstore...")
        vectorstore = build_vectorstore()
        with open(MTIME_RECORD, "wb") as f:
            pickle.dump(current_mtimes, f)
        return vectorstore

    print("✅ Vectorstore masih valid.")
    return load_vectorstore()


# MAIN LOOP
vectorstore = auto_update_vectorstore_with_mtime()
    
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

rag_chain = (
    RunnableParallel({
        "context": lambda x: retriever.invoke(x["input"]),
        "question": lambda x: x["input"],    
        "history": lambda x: memory.buffer      
    })
    | rag_prompt
    | llm
    | StrOutputParser()
)

load_chat_history_to_memory()

exit_commands = ["stop", "quit", "exit", "end conversation"]

while True:
    audio, fs = rekam_dengan_postbuffer(durasi=4, post_buffer=1)
    if audio is None:
        continue
    speech_timestamps = get_speech_timestamps(audio, model, sampling_rate=fs)

    if len(speech_timestamps) == 0:
        print("❌ Tidak ada suara, skip turn.")
        continue

    # Simpan rekaman
    file_wav = os.path.join(TEMP_DIR, "rekaman.wav")
    simpan_wav(file_wav, audio, fs)

    teks_input = stt_whisper(file_wav)
    if teks_input.strip() == "":
        msg = "I'm sorry, can't hear you."
        print("Cihaku :", msg)
        tts_coqui_simpleaudio(msg)
        continue

    print(f"📝 Teks dari suara: {teks_input}")
    if any(cmd in teks_input.lower() for cmd in exit_commands):
        msgstop = "Okay then, goodbye."
        print("\nCihaku :", msgstop)
        tts_coqui_simpleaudio(msgstop)
        break
    
    jawaban_ai = chatAI_stream_and_tts(teks_input)

    if jawaban_ai.lower().startswith("Chatbot:"):
        jawaban_ai = jawaban_ai[len("Chatbot:"):].strip() 
    
    logging.info(f"Input: {teks_input}")
    logging.info(f"AI Output: {jawaban_ai}")

    save_chat_history(teks_input, jawaban_ai)
    
    memory.chat_memory.add_user_message(teks_input)
    
    memory.chat_memory.add_ai_message(jawaban_ai)
    
    if len(memory.chat_memory.messages) > 20:
        memory.chat_memory.messages = memory.chat_memory.messages[-20:]