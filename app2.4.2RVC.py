import sys
sys.path.extend([
    "D:/VisualSC/MyChatbotRAG",
    "D:/VisualSC/MyChatbotRAG/tools/rvcmodels"
])

import os
import json
import asyncio
import logging
import numpy as np
import torch
import sounddevice as sd
import soundfile as sf
import librosa
import edge_tts
import whisper
from pydub import AudioSegment
from pathlib import Path
import simpleaudio as sa
from fairseq import checkpoint_utils
from tools.rvcmodels.infer_pack.models import SynthesizerTrnMs256NSFsid, SynthesizerTrnMs256NSFsid_nono
from tools.rvcmodels.vc_infer_pipeline import VC
from langchain_ollama import OllamaLLM
from configlama import OLLAMA_API_URL
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

MODEL_DIR = "D:/VisualSC/MyChatbotRAG/tools/rvcmodels/weights/ayaka-jp"
MODEL_PTH = os.path.join(MODEL_DIR, "ayaka-jp.pth")
MODEL_INFO = os.path.join(MODEL_DIR, "D:/VisualSC/MyChatbotRAG/tools/rvcmodels/weights/model_info.json")
CONFIG_PATH = os.path.join(MODEL_DIR, "config.json")
INDEX_PATH = os.path.join(MODEL_DIR, "added_IVF1830_Flat_nprobe_9.index")
FEATURE_NPY_PATH = os.path.join(MODEL_DIR, "total_fea.npy")
HUBERT_PATH = "D:/VisualSC/MyChatbotRAG/tools/rvcmodels/hubert_base.pt"

# Constants
TEMP_DIR = Path("D:/VisualSC/MyChatbotRAG/TempPython")
os.makedirs(TEMP_DIR, exist_ok=True)

# Initialize logging
logging.basicConfig(
    level=logging.INFO,
    filename='D:/VisualSC/MyChatbotRAG/chatbot_log.txt',
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def check_output_sample_rate(path):
    """Periksa sample rate dari file WAV output"""
    try:
        _, sr = sf.read(str(path))
        print(f"✅ Sample rate hasil konversi RVC: {sr} Hz")
    except Exception as e:
        print(f"❌ Gagal membaca sample rate: {e}")


class RVCModel:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.is_half = False
        self.initialize_models()
        self.vc_pipeline = VC(self.tgt_sr, self.device, is_half=self.is_half)

    def initialize_models(self):
        """Initialize all RVC-related models"""
        # HuBERT model
        hubert_path = "D:/VisualSC/MyChatbotRAG/tools/rvcmodels/hubert_base.pt"
        self.hubert_model = self.load_hubert_model(hubert_path)
        
        # RVC model
        model_dir = "D:/VisualSC/MyChatbotRAG/tools/multi_model_rvc/weights/model_pack/Ayaka"
        self.load_rvc_model(model_dir)

    def load_hubert_model(self, hubert_path):
        """Load HuBERT model for feature extraction"""
        models, _, _ = checkpoint_utils.load_model_ensemble_and_task([hubert_path])
        model = models[0]
        model.to(self.device).eval()
        return model

    def load_rvc_model(self, model_dir):
        cpt = torch.load(MODEL_PTH, map_location="cpu")
        config = cpt["config"]
        config[-3] = cpt["weight"]["emb_g.weight"].shape[0]
        self.tgt_sr = config[-1]
        self.if_f0 = cpt.get("f0", 1)

        if self.if_f0 == 1:
            self.net_g = SynthesizerTrnMs256NSFsid(*config, is_half=self.is_half)
        else:
            self.net_g = SynthesizerTrnMs256NSFsid_nono(*config)

        del self.net_g.enc_q
        self.net_g.load_state_dict(cpt["weight"], strict=False)
        self.net_g.eval().to(self.device)
        self.net_g = self.net_g.half() if self.is_half else self.net_g.float()

    async def convert_voice(self, input_audio_path, output_path):
        """Convert voice using RVC pipeline"""
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
            file_index=os.path.join(INDEX_PATH),
            file_big_npy=os.path.join(MODEL_DIR, "total_fea.npy"),
            index_rate=1,
            if_f0=self.if_f0,
            f0_file=None
        )

        sf.write(str(output_path), out_audio, self.tgt_sr)

        return output_path

class VoiceChat:
    def __init__(self):
        self.rvc = RVCModel()
        self.initialize_components()
        
    def initialize_components(self):
        """Initialize all voice chat components"""
        # Initialize Whisper
        self.whisper_model = whisper.load_model("small").to(self.rvc.device)
        
        # Initialize VAD
        self.vad_model, utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            trust_repo=True
        )
        self.get_speech_timestamps = utils[0]
        
        # Initialize LLM
        self.llm = OllamaLLM(
            model="phi3:mini",
            base_url="http://localhost:11434",
            streaming=True
        )
        
        # Initialize memory and prompt
        self.memory = ConversationBufferMemory(return_messages=True)
        self.rag_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a friendly AI assistant..."),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{question}")
        ])

    async def record_audio(self, duration=5, sample_rate=16000):
        """Record audio with VAD"""
        print(f"Recording for {duration} seconds...")
        audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
        sd.wait()
        
        # Detect speech segments
        speech = self.get_speech_timestamps(
            np.squeeze(audio), 
            self.vad_model, 
            sampling_rate=sample_rate
        )
        
        return (np.squeeze(audio[speech[0]['start']:speech[-1]['end']]), 
                sample_rate) if speech else (None, sample_rate)

    async def process_voice_input(self):
        """Full voice input processing pipeline"""
        # Record audio
        audio, sr = await self.record_audio()
        if audio is None:
            print("🔇 No speech detected")
            return None
            
        # Save and transcribe
        input_path = TEMP_DIR / "input.wav"
        sf.write(input_path, audio, sr)
        text = await self.transcribe(input_path)
        
        # Generate response
        if text:
            response = await self.generate_response(text)
            await self.text_to_speech(response)
            return response
        return None

    async def transcribe(self, audio_path):
        """Transcribe audio using Whisper"""
        try:
            result = self.whisper_model.transcribe(str(audio_path), language="en")
            return result["text"].strip()
        except Exception as e:
            logging.error(f"Transcription failed: {str(e)}")
            return ""

    async def generate_response(self, text):
        """Generate LLM response"""
        # Your existing RAG chain implementation
        try:
            chain = self.rag_prompt | self.llm
            response = await chain.ainvoke({"question": text, "history": self.memory.buffer})
            
            # Update memory
            self.memory.save_context({"input": text}, {"output": response})
            return response
        except Exception as e:
            logging.error(f"LLM response failed: {str(e)}")
            return "I couldn't generate a response."


    async def text_to_speech(self, text, voice="en-US-AnaNeural"):
        """Convert text to speech with RVC conversion"""
        if not text or not isinstance(text, str):
            logging.error("Invalid text for TTS")
            return False
            
        try:
            # Generate TTS
            mp3_path = TEMP_DIR / "tts_temp.mp3"
            await edge_tts.Communicate(text, voice).save(str(mp3_path))
            
            # Convert with RVC
            wav_path = TEMP_DIR / "output.wav"
            success = await self.rvc.convert_voice(mp3_path, wav_path)
            
            check_output_sample_rate(wav_path)
            
            # Play audio if successful
            if success:
                self.play_audio(wav_path)
            
            # Cleanup
            mp3_path.unlink(missing_ok=True)
            wav_path.unlink(missing_ok=True)
            return success
            
        except Exception as e:
            logging.error(f"TTS+RVC failed: {str(e)}")
            return False

    def play_audio(self, path):
        """Play audio file"""
        try:
            audio = AudioSegment.from_file(path)
            samples = np.array(audio.get_array_of_samples())
            sa.play_buffer(
                (samples * 32767).astype(np.int16),
                1,
                2,
                audio.frame_rate
            ).wait_done()
        except Exception as e:
            logging.error(f"Audio playback failed: {str(e)}")

    async def run(self):
        """Main voice chat loop"""
        print("🚀 Voice chat system ready. Press Ctrl+C to exit.")
        try:
            while True:
                await self.process_voice_input()
        except KeyboardInterrupt:
            print("\n🛑 Shutting down...")
        finally:
            # Cleanup
            for file in TEMP_DIR.glob("*"):
                try:
                    file.unlink()
                except:
                    pass

if __name__ == "__main__":
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    chat = VoiceChat()
    asyncio.run(chat.run())
    
