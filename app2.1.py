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

logging.basicConfig(level=logging.INFO, filename='D:/VisualSC/MyChatbotRAG/chatbot_log.txt', filemode='a', format='%(asctime)s - %(levelname)s - %(message)s')

urls = ["https://en.wikipedia.org/wiki/AI", "https://en.wikipedia.org/wiki/NLP"]
loader = WebBaseLoader("urls")
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

#vector retriever


# Temp folder
TEMP_DIR = "D:/VisualSC/MyChatbotRAG/TempPython"
os.makedirs(TEMP_DIR, exist_ok=True)
warnings.filterwarnings("ignore", category=UserWarning)

# LLM Model (Ollama)
llm = OllamaLLM(
    model="llama2:7b-chat-q4_0",
    base_url="http://localhost:11434",
    streaming = True,
    callbacks=[StreamingStdOutCallbackHandler()]
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

llm_chain = prompt | llm | StrOutputParser()

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
        
    splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 50)
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

def pitcher(input_file, pitch=0.2):
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

def stt_whisper(audio_file):
    model = whisper.load_model("small").to("cuda")
    result = model.transcribe(audio_file, language='en')
    return result["text"]

def chatAI(user_input):
    return rag_chain.invoke({"input": user_input})

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

def filter_words(teks):
    bad_words = ["badword1", "badword2"]
    for word in bad_words:
        teks = teks.replace(word, "***")
    return teks


# MAIN LOOP

# Personality
rag_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a friendly, helpful assistant speaking in casual tone. Use context below if available:\n\n{context}"),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{question}")
])

if os.path.exists(os.path.join(VECTOR_STORE_DIR, "index")):
    vectorstore = load_vectorstore()
else:
    vectorstore = build_vectorstore()
    
retriever = vectorstore.as_retriever()

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
        print("❌ Tidak ada teks yang dikenali dari suara, skip turn.")
        continue

    print(f"📝 Teks dari suara: {teks_input}")
    if any(cmd in teks_input.lower() for cmd in exit_commands):
        print("\n🚪 Perintah keluar terdeteksi. Menghentikan chatbot...")
        break

    jawaban_ai = chatAI(teks_input)
    
    if jawaban_ai.lower().startswith("chatbot:"):
        jawaban_ai = jawaban_ai[len("chatbot:"):].strip()
        
    jawaban_ai = re.sub(r"\*(.+?)\*", "", jawaban_ai, flags=re.DOTALL)
    jawaban_ai = re.sub(r"\s+", " ", jawaban_ai).strip()    
    
    logging.info(f"Input: {teks_input}")
    logging.info(f"AI Output: {jawaban_ai}")

    
    print(f"Jawaban AI: {jawaban_ai}")
    
    save_chat_history(teks_input, jawaban_ai)
    
    memory.chat_memory.add_user_message(teks_input)
    
    memory.chat_memory.add_ai_message(jawaban_ai)
    
    if len(memory.chat_memory.messages) > 20:
        memory.chat_memory.messages = memory.chat_memory.messages[-20:]


    tts_coqui_simpleaudio(jawaban_ai)
