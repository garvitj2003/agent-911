import asyncio
import io
import wave
import numpy as np
from groq import Groq

class SpeechAnalyzer:
    def __init__(self, api_key: str, model_name: str = "whisper-large-v3-turbo"):
        self.client = Groq(api_key=api_key)
        self.model_name = model_name

    async def transcribe(self, audio_data: np.ndarray) -> str:
        """Transcribes raw numpy float32 audio data to text using Groq's Whisper API."""
        if audio_data is None or len(audio_data) == 0:
            return ""
            
        # Convert float32 numpy array to 16-bit PCM WAV bytes in-memory
        audio_clamped = np.clip(audio_data, -1.0, 1.0)
        audio_int16 = (audio_clamped * 32767).astype(np.int16)
        
        wav_io = io.BytesIO()
        with wave.open(wav_io, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)      # 16-bit PCM
            wf.setframerate(16000)   # Matching AudioManager default rate
            wf.writeframes(audio_int16.tobytes())
            
        wav_bytes = wav_io.getvalue()
        
        loop = asyncio.get_running_loop()
        def _transcribe():
            try:
                # Transcribe from wav bytes using Groq SDK
                transcription = self.client.audio.transcriptions.create(
                    file=("audio.wav", wav_bytes),
                    model=self.model_name,
                    response_format="json"
                )
                return transcription.text.strip()
            except Exception as e:
                return f"[Groq STT failed: {e}]"
                
        return await loop.run_in_executor(None, _transcribe)
