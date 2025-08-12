<<<<<<< HEAD
import sys
sys.path.extend([
    "tools/rvcmodels",
    ""
])
import gradio as gr
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
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
from pydub import AudioSegment
from langchain.schema import AIMessage, HumanMessage
from pathlib import Path
from scipy.io.wavfile import write as wav_write
from silero_vad import VADIterator, get_speech_timestamps, collect_chunks
from tools.rvcmodels.infer_pack.models import SynthesizerTrnMs256NSFsid, SynthesizerTrnMs256NSFsid_nono
from tools.rvcmodels.vc_infer_pipeline import VC
from fairseq import checkpoint_utils
import edge_tts
import whisper
import torch
import asyncio
import sounddevice as sd
import soundfile as sf
import numpy as np
import librosa
import pickle
import logging
import re
import unstructured
import json
import simpleaudio as sa

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("fairseq").setLevel(logging.WARNING)

def load_chat_history_json():
    if not os.path.exists(CHAT_HISTORY_PATH):
        return []

    with open(CHAT_HISTORY_PATH, "r") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return []

def parse_chat_messages(history_data):
    parsed = []
    for msg in history_data:
        role = msg.get("role")
        content = msg.get("content")
        if not content:
            continue
        if role == "human":
            parsed.append(HumanMessage(content=content))
        elif role == "ai":
            parsed.append(AIMessage(content=content))
    return parsed

def save_chat_history_json(messages):
    data = []
    for m in messages:
        if isinstance(m, HumanMessage):
            data.append({"role": "human", "content": m.content})
        elif isinstance(m, AIMessage):
            data.append({"role": "ai", "content": m.content})
    with open(CHAT_HISTORY_PATH, "w") as f:
        json.dump(data, f, indent=2)

MODEL_INFO = "tools/rvcmodels/weights/model_info.index"
MODEL_DIR = "tools/rvcmodels/weights/ayaka-jp"
MODEL_PATH = os.path.join(MODEL_DIR, "ayaka-jp.pth")
INDEX_PATH = os.path.join(MODEL_DIR, "added_IVF1830_Flat_nprobe_9.index")
FEATURE_NPY_PATH = os.path.join(MODEL_DIR, "total_fea.npy")
HUBERT_PATH = "tools/rvcmodels/hubert_base.pt"

CHAT_HISTORY_PATH = "chat_history/chat_log.json"

TEMP_DIR = Path("TempPython")
os.makedirs(TEMP_DIR, exist_ok=True)

class RVCModel:
    def __init__(self):
        self.device = "cpu"
        self.is_half = False
        self.initialize_models()
        self.vc_pipeline = VC(self.tgr_sr, self.device, is_half=False)
        
    def initialize_models(self):
        self.hubert_model = self.load_hubert_model(HUBERT_PATH)
        self.load_rvc_model(MODEL_DIR)
        
    def load_hubert_model(self, hubert_path):
        models, _, _ = checkpoint_utils.load_model_ensemble_and_task([HUBERT_PATH])
        model = models[0]
        model.to(self.device).eval()
        return model
    
    def load_rvc_model(self, MODEL_DIR):
        cpt = torch.load(MODEL_PATH, map_location="cpu")
        config = cpt["config"]
        config[-3] = cpt["weight"]["emb_g.weight"].shape[0]
        self.tgr_sr = config[-1]
        self.if_f0 = cpt.get("f0", 1)
        
        if self.if_f0 == 1 :
            self.net_g = SynthesizerTrnMs256NSFsid(*config, is_half=self.is_half)
        else:
            self.net_g = SynthesizerTrnMs256NSFsid_nono(*config)
        
        del self.net_g.enc_q
        self.net_g.load_state_dict(cpt["weight"], strict=False)
        self.net_g.eval().to(self.device)
        self.net_g = self.net_g.half()
        
        if self.is_half :
            self.net_g = self.net_g.half()
        else:
            self.net_g = self.net_g.float()
        
    async def convert_voice(self, input_audio_path, output_path):
        audio, sr = sf.read(input_audio_path)
        if len(audio.shape) == 2:
            audio = np.mean(audio, axis=1)
            
        out_audio = self.vc_pipeline.pipeline(
            model=self.hubert_model,
            net_g=self.net_g,
            sid=0,
            audio=audio,
            times=[0, 0, 0],
            f0_up_key=0,
            f0_method="pm",
            file_index=str(INDEX_PATH),
            file_big_npy=str(FEATURE_NPY_PATH),
            index_rate=1,
            if_f0=self.if_f0,
            f0_file=None
        )
        
        sf.write(str(output_path), out_audio, self.tgr_sr)
        return output_path
    
class VoiceChat:
    def __init__(self):
        self.rvc = RVCModel()
        self.initialize_component()
        
    def initialize_component(self):
        self.whisper_model = whisper.load_model("small").to(self.rvc.device)
        self.vad_model, utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            trust_repo=True
        )
        self.get_speech_timestamps = utils[0]
        
        self.llm = OllamaLLM(
            model="phi3:mini",
            base_url="http://localhost:11434",
            streaming=True
        )

        self.memory = ConversationBufferMemory(return_messages=True)
        
        history_data = load_chat_history_json()
        parsed_history = parse_chat_messages(history_data)
        self.memory.chat_memory.messages = parsed_history
        
        self.rag_prompt = ChatPromptTemplate.from_messages([
            ("system", 
            "You are Chikaku, a modest and emotionally aware"
            "You do not engage to use emoticon. "
            "You assist users with honesty, empathy, and clarity. "
            "You can understand and express emotional nuance, but always remain grounded as an AI. "
            "Be helpful, warm, and respectful, but never simulate imaginary roles or personas."
            "Always respond in english, preferred Bahasa Indonesia"
            "jawab secara singkat saja"
            ),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{question}")
        ])
        
    async def run_text_mode(self):
        print("Mode input teks aktif. Ketik 'exit' untuk keluar.")
        try:
            while True:
                user_input = input("Kamu: ")
                if user_input.lower() in ["exit", "quit"]:
                    print("Keluar dari mode teks.")
                    break
                response = await self.generate_response(user_input)
                print(f"Chikaku: {response}")
                await self.text_to_speech(response)
        except KeyboardInterrupt:
            print("\nDihentikan.")

        
    async def record_audio(self, duration=5, sample_rate=16000):
        print(f"Merekam suara selama {duration} detik")
        audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype="float32")
        sd.wait()
        speech = self.get_speech_timestamps(np.squeeze(audio), self.vad_model, sampling_rate=sample_rate)
        return (np.squeeze(audio[speech[0]['start']:speech[-1]['end']]), sample_rate) if speech else (None, sample_rate)
        
    async def process_voice_input(self):
        audio, sr = await self.record_audio()
        if audio is None:
            print("Tidak mendeteksi suara")
            return None
        
        input_path = TEMP_DIR / "input.wav"
        sf.write(input_path, audio, sr)
        text = await self.transcribe(input_path)
        
        if text:
            response = await self.generate_response(text)
            await self.text_to_speech(response)
            return response
        return None
        
    async def transcribe(self, audio_path):
        try:
            result = self.whisper_model.transcribe(str(audio_path), language="en")
            return result["text"].strip()
        except Exception as e:
            logging.error(f"Gagal transkripsi: {str(e)}")
            return ""
        
    async def generate_response(self, text):
        try:
            chain = self.rag_prompt | self.llm
            response = await chain.ainvoke({"question": text, "history": self.memory.buffer})
            self.memory.save_context({"input": text}, {"output": response})
            save_chat_history_json(self.memory.chat_memory.messages)
            return response
        except Exception as e:
            logging.error(f"LLM gagal: {str(e)}")
            return "Maaf, saya tidak bisa menjawab."

    async def text_to_speech(self, text, voice="en-US-AnaNeural"):
            if not text or not isinstance(text, str):
                logging.error("Input tidak valid untuk TTS")
                return False
            try:
                mp3_path = TEMP_DIR / "tts_temp.mp3"
                await edge_tts.Communicate(text, voice).save(str(mp3_path))
                wav_path = TEMP_DIR / "output.wav"
                success = await self.rvc.convert_voice(mp3_path, wav_path)
                if success:
                    self.play_audio(wav_path)
                mp3_path.unlink(missing_ok=True)
                wav_path.unlink(missing_ok=True)
                return success
            except Exception as e:
                logging.error(f"TTS+RVC gagal: {str(e)}")
                return False
            
    def play_audio(self, path):
        try:
            audio = AudioSegment.from_file(path)
            samples = np.array(audio.get_array_of_samples())
            if audio.frame_rate != 44100:
                samples = samples.astype(np.float32) / 32768.0
                samples_resampled = librosa.resample(samples, orig_sr=audio.frame_rate, target_sr=44100)
                samples = (samples_resampled * 32767).astype(np.int16)
                sample_rate = 44100
            else:
                samples = (samples * 32767).astype(np.int16)
                sample_rate = audio.frame_rate

            sa.play_buffer(samples, 1, 2, sample_rate).wait_done()
        except Exception as e:
            logging.error(f"Gagal memutar audio: {str(e)}")
        
    async def run(self):
        print("Voice chat aktif. CTRL+C untuk keluar")
        try:
            while True:
                await self.process_voice_input()
        except KeyboardInterrupt:
            print("Voice chat dihentikan.")
        finally:
            for file in TEMP_DIR.glob("*"):
                try:
                    file.unlink()
                except:
                    pass

def run_async(func):
    try:
        asyncio.run(func())
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(func())
                
chat_instance = None

async def chat_with_voice_or_text(input_text=None, input_audio=None, voice_mode="Edge Only"):
    global chat_instance
    if chat_instance is None:
        chat_instance = VoiceChat()

    if input_audio:
        sf.write("temp_input.wav", input_audio[1], input_audio[0])
        text = await chat_instance.transcribe("temp_input.wav")
    else:
        text = input_text

    if not text:
        return "No input received", None

    response = await chat_instance.generate_response(text)

    mp3_path = TEMP_DIR / "gradio_tts.mp3"
    wav_path = TEMP_DIR / "gradio_rvc.wav"

    try:
        await edge_tts.Communicate(response, voice="en-US-AnaNeural").save(str(mp3_path))

        if voice_mode == "Edge + RVC":
            await chat_instance.rvc.convert_voice(mp3_path, wav_path)
            return response, str(wav_path)
        else:
            return response, str(mp3_path)

    except Exception as e:
        logging.error(f"Gagal TTS/RVC: {e}")
        return "Maaf, tidak bisa membuat audio.", None



if __name__ == "__main__":
    print("Pilih mode:")
    print("1. Voice chat (terminal)")
    print("2. Text input (terminal)")
    print("3. Web UI via Gradio")

    try:
        choice = input("Masukkan pilihan (1/2/3): ").strip()
        if choice == "1":
            chat_instance = VoiceChat()
            asyncio.run(chat_instance.run())
        elif choice == "2":
            chat_instance = VoiceChat()
            asyncio.run(chat_instance.run_text_mode())
        elif choice == "3":

            with gr.Blocks() as demo:
                gr.Markdown("### Chikaku AI Assistant (Text & Voice)")

                with gr.Row():
                    txt_input = gr.Textbox(label="Your message")
                    audio_input = gr.Audio(label="Or speak", type="numpy", format="wav")

                with gr.Row():
                    voice_selector = gr.Radio(
                        choices=["Edge Only", "Edge + RVC"],
                        value="Edge Only",
                        label="Voice Output Mode"
                    )

                with gr.Row():
                    submit_btn = gr.Button("Send")

                txt_output = gr.Textbox(label="Chikaku says")
                audio_output = gr.Audio(label="Voice Output")

                submit_btn.click(
                    fn=chat_with_voice_or_text,
                    inputs=[txt_input, audio_input, voice_selector],
                    outputs=[txt_output, audio_output]
                )


            demo.launch(server_name="0.0.0.0", server_port=7860)
        else:
            print("Pilihan tidak valid.")
    except KeyboardInterrupt:
        print("\nDihentikan.")
=======
import sys
sys.path.extend([
    "tools/rvcmodels",
    ""
])
import gradio as gr
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
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
from pydub import AudioSegment
from langchain.schema import AIMessage, HumanMessage
from pathlib import Path
from scipy.io.wavfile import write as wav_write
from silero_vad import VADIterator, get_speech_timestamps, collect_chunks
from tools.rvcmodels.infer_pack.models import SynthesizerTrnMs256NSFsid, SynthesizerTrnMs256NSFsid_nono
from tools.rvcmodels.vc_infer_pipeline import VC
from fairseq import checkpoint_utils
import edge_tts
import whisper
import torch
import asyncio
import sounddevice as sd
import soundfile as sf
import numpy as np
import librosa
import pickle
import logging
import re
import unstructured
import json
import simpleaudio as sa

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("fairseq").setLevel(logging.WARNING)

def load_chat_history_json():
    if not os.path.exists(CHAT_HISTORY_PATH):
        return []

    with open(CHAT_HISTORY_PATH, "r") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return []

def parse_chat_messages(history_data):
    parsed = []
    for msg in history_data:
        role = msg.get("role")
        content = msg.get("content")
        if not content:
            continue
        if role == "human":
            parsed.append(HumanMessage(content=content))
        elif role == "ai":
            parsed.append(AIMessage(content=content))
    return parsed

def save_chat_history_json(messages):
    data = []
    for m in messages:
        if isinstance(m, HumanMessage):
            data.append({"role": "human", "content": m.content})
        elif isinstance(m, AIMessage):
            data.append({"role": "ai", "content": m.content})
    with open(CHAT_HISTORY_PATH, "w") as f:
        json.dump(data, f, indent=2)

MODEL_INFO = "tools/rvcmodels/weights/model_info.index"
MODEL_DIR = "tools/rvcmodels/weights/ayaka-jp"
MODEL_PATH = os.path.join(MODEL_DIR, "ayaka-jp.pth")
INDEX_PATH = os.path.join(MODEL_DIR, "added_IVF1830_Flat_nprobe_9.index")
FEATURE_NPY_PATH = os.path.join(MODEL_DIR, "total_fea.npy")
HUBERT_PATH = "tools/rvcmodels/hubert_base.pt"

CHAT_HISTORY_PATH = "chat_history/chat_log.json"

TEMP_DIR = Path("TempPython")
os.makedirs(TEMP_DIR, exist_ok=True)

class RVCModel:
    def __init__(self):
        self.device = "cpu"
        self.is_half = False
        self.initialize_models()
        self.vc_pipeline = VC(self.tgr_sr, self.device, is_half=False)
        
    def initialize_models(self):
        self.hubert_model = self.load_hubert_model(HUBERT_PATH)
        self.load_rvc_model(MODEL_DIR)
        
    def load_hubert_model(self, hubert_path):
        models, _, _ = checkpoint_utils.load_model_ensemble_and_task([HUBERT_PATH])
        model = models[0]
        model.to(self.device).eval()
        return model
    
    def load_rvc_model(self, MODEL_DIR):
        cpt = torch.load(MODEL_PATH, map_location="cpu")
        config = cpt["config"]
        config[-3] = cpt["weight"]["emb_g.weight"].shape[0]
        self.tgr_sr = config[-1]
        self.if_f0 = cpt.get("f0", 1)
        
        if self.if_f0 == 1 :
            self.net_g = SynthesizerTrnMs256NSFsid(*config, is_half=self.is_half)
        else:
            self.net_g = SynthesizerTrnMs256NSFsid_nono(*config)
        
        del self.net_g.enc_q
        self.net_g.load_state_dict(cpt["weight"], strict=False)
        self.net_g.eval().to(self.device)
        self.net_g = self.net_g.half()
        
        if self.is_half :
            self.net_g = self.net_g.half()
        else:
            self.net_g = self.net_g.float()
        
    async def convert_voice(self, input_audio_path, output_path):
        audio, sr = sf.read(input_audio_path)
        if len(audio.shape) == 2:
            audio = np.mean(audio, axis=1)
            
        out_audio = self.vc_pipeline.pipeline(
            model=self.hubert_model,
            net_g=self.net_g,
            sid=0,
            audio=audio,
            times=[0, 0, 0],
            f0_up_key=0,
            f0_method="pm",
            file_index=str(INDEX_PATH),
            file_big_npy=str(FEATURE_NPY_PATH),
            index_rate=1,
            if_f0=self.if_f0,
            f0_file=None
        )
        
        sf.write(str(output_path), out_audio, self.tgr_sr)
        return output_path
    
class VoiceChat:
    def __init__(self):
        self.rvc = RVCModel()
        self.initialize_component()
        
    def initialize_component(self):
        self.whisper_model = whisper.load_model("small").to(self.rvc.device)
        self.vad_model, utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            trust_repo=True
        )
        self.get_speech_timestamps = utils[0]
        
        self.llm = OllamaLLM(
            model="phi3:mini",
            base_url="http://localhost:11434",
            streaming=True
        )

        self.memory = ConversationBufferMemory(return_messages=True)
        
        history_data = load_chat_history_json()
        parsed_history = parse_chat_messages(history_data)
        self.memory.chat_memory.messages = parsed_history
        
        self.rag_prompt = ChatPromptTemplate.from_messages([
            ("system", 
            "You are Chikaku, a modest and emotionally aware"
            "You do not engage to use emoticon. "
            "You assist users with honesty, empathy, and clarity. "
            "You can understand and express emotional nuance, but always remain grounded as an AI. "
            "Be helpful, warm, and respectful, but never simulate imaginary roles or personas."
            "Always respond in english, preferred Bahasa Indonesia"
            "jawab secara singkat saja"
            ),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{question}")
        ])
        
    async def run_text_mode(self):
        print("Mode input teks aktif. Ketik 'exit' untuk keluar.")
        try:
            while True:
                user_input = input("Kamu: ")
                if user_input.lower() in ["exit", "quit"]:
                    print("Keluar dari mode teks.")
                    break
                response = await self.generate_response(user_input)
                print(f"Chikaku: {response}")
                await self.text_to_speech(response)
        except KeyboardInterrupt:
            print("\nDihentikan.")

        
    async def record_audio(self, duration=5, sample_rate=16000):
        print(f"Merekam suara selama {duration} detik")
        audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype="float32")
        sd.wait()
        speech = self.get_speech_timestamps(np.squeeze(audio), self.vad_model, sampling_rate=sample_rate)
        return (np.squeeze(audio[speech[0]['start']:speech[-1]['end']]), sample_rate) if speech else (None, sample_rate)
        
    async def process_voice_input(self):
        audio, sr = await self.record_audio()
        if audio is None:
            print("Tidak mendeteksi suara")
            return None
        
        input_path = TEMP_DIR / "input.wav"
        sf.write(input_path, audio, sr)
        text = await self.transcribe(input_path)
        
        if text:
            response = await self.generate_response(text)
            await self.text_to_speech(response)
            return response
        return None
        
    async def transcribe(self, audio_path):
        try:
            result = self.whisper_model.transcribe(str(audio_path), language="en")
            return result["text"].strip()
        except Exception as e:
            logging.error(f"Gagal transkripsi: {str(e)}")
            return ""
        
    async def generate_response(self, text):
        try:
            chain = self.rag_prompt | self.llm
            response = await chain.ainvoke({"question": text, "history": self.memory.buffer})
            self.memory.save_context({"input": text}, {"output": response})
            save_chat_history_json(self.memory.chat_memory.messages)
            return response
        except Exception as e:
            logging.error(f"LLM gagal: {str(e)}")
            return "Maaf, saya tidak bisa menjawab."

    async def text_to_speech(self, text, voice="en-US-AnaNeural"):
            if not text or not isinstance(text, str):
                logging.error("Input tidak valid untuk TTS")
                return False
            try:
                mp3_path = TEMP_DIR / "tts_temp.mp3"
                await edge_tts.Communicate(text, voice).save(str(mp3_path))
                wav_path = TEMP_DIR / "output.wav"
                success = await self.rvc.convert_voice(mp3_path, wav_path)
                if success:
                    self.play_audio(wav_path)
                mp3_path.unlink(missing_ok=True)
                wav_path.unlink(missing_ok=True)
                return success
            except Exception as e:
                logging.error(f"TTS+RVC gagal: {str(e)}")
                return False
            
    def play_audio(self, path):
        try:
            audio = AudioSegment.from_file(path)
            samples = np.array(audio.get_array_of_samples())
            if audio.frame_rate != 44100:
                samples = samples.astype(np.float32) / 32768.0
                samples_resampled = librosa.resample(samples, orig_sr=audio.frame_rate, target_sr=44100)
                samples = (samples_resampled * 32767).astype(np.int16)
                sample_rate = 44100
            else:
                samples = (samples * 32767).astype(np.int16)
                sample_rate = audio.frame_rate

            sa.play_buffer(samples, 1, 2, sample_rate).wait_done()
        except Exception as e:
            logging.error(f"Gagal memutar audio: {str(e)}")
        
    async def run(self):
        print("Voice chat aktif. CTRL+C untuk keluar")
        try:
            while True:
                await self.process_voice_input()
        except KeyboardInterrupt:
            print("Voice chat dihentikan.")
        finally:
            for file in TEMP_DIR.glob("*"):
                try:
                    file.unlink()
                except:
                    pass

def run_async(func):
    try:
        asyncio.run(func())
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(func())
                
chat_instance = None

async def chat_with_voice_or_text(input_text=None, input_audio=None, voice_mode="Edge Only"):
    global chat_instance
    if chat_instance is None:
        chat_instance = VoiceChat()

    if input_audio:
        sf.write("temp_input.wav", input_audio[1], input_audio[0])
        text = await chat_instance.transcribe("temp_input.wav")
    else:
        text = input_text

    if not text:
        return "No input received", None

    response = await chat_instance.generate_response(text)

    mp3_path = TEMP_DIR / "gradio_tts.mp3"
    wav_path = TEMP_DIR / "gradio_rvc.wav"

    try:
        await edge_tts.Communicate(response, voice="en-US-AnaNeural").save(str(mp3_path))

        if voice_mode == "Edge + RVC":
            await chat_instance.rvc.convert_voice(mp3_path, wav_path)
            return response, str(wav_path)
        else:
            return response, str(mp3_path)

    except Exception as e:
        logging.error(f"Gagal TTS/RVC: {e}")
        return "Maaf, tidak bisa membuat audio.", None



if __name__ == "__main__":
    print("Pilih mode:")
    print("1. Voice chat (terminal)")
    print("2. Text input (terminal)")
    print("3. Web UI via Gradio")

    try:
        choice = input("Masukkan pilihan (1/2/3): ").strip()
        if choice == "1":
            chat_instance = VoiceChat()
            asyncio.run(chat_instance.run())
        elif choice == "2":
            chat_instance = VoiceChat()
            asyncio.run(chat_instance.run_text_mode())
        elif choice == "3":

            with gr.Blocks() as demo:
                gr.Markdown("### Chikaku AI Assistant (Text & Voice)")

                with gr.Row():
                    txt_input = gr.Textbox(label="Your message")
                    audio_input = gr.Audio(label="Or speak", type="numpy", format="wav")

                with gr.Row():
                    voice_selector = gr.Radio(
                        choices=["Edge Only", "Edge + RVC"],
                        value="Edge Only",
                        label="Voice Output Mode"
                    )

                with gr.Row():
                    submit_btn = gr.Button("Send")

                txt_output = gr.Textbox(label="Chikaku says")
                audio_output = gr.Audio(label="Voice Output")

                submit_btn.click(
                    fn=chat_with_voice_or_text,
                    inputs=[txt_input, audio_input, voice_selector],
                    outputs=[txt_output, audio_output]
                )


            demo.launch(server_name="0.0.0.0", server_port=7860)
        else:
            print("Pilihan tidak valid.")
    except KeyboardInterrupt:
        print("\nDihentikan.")
>>>>>>> 46f8efefacdbcef7e0ed266c41059ff89321cbc6
