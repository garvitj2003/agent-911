import asyncio
import numpy as np
import sounddevice as sd
import subprocess
import tempfile
import os
import time

class AudioManager:
    def __init__(self, api_key: str, voice_name: str = "pNInz6obpgDQGcFmaJgB", sample_rate=16000, silence_timeout=1.5, min_speech_duration=0.2, max_speech_duration=15.0):
        self.api_key = api_key
        self.voice_name = voice_name
        self.sample_rate = sample_rate
        self.silence_timeout = silence_timeout
        self.min_speech_duration = min_speech_duration
        self.max_speech_duration = max_speech_duration
        self.background_noise = 0.01  # Initial fallback value
        self.is_recording = False
        self.is_prod = os.getenv("PROD", "true").strip().lower() != "false"

    async def speak(self, text: str, error_callback=None):
        """Generates TTS audio using ElevenLabs API and plays it via mpv in the background."""
        if not self.is_prod:
            return
            
        loop = asyncio.get_running_loop()
        def _speak():
            import urllib.request
            import json
            
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
                temp_path = f.name
            try:
                # Post request to ElevenLabs Text-to-Speech API
                url = f"https://api.elevenlabs.io/v1/text-to-speech/{self.voice_name}"
                headers = {
                    "xi-api-key": self.api_key,
                    "Content-Type": "application/json"
                }
                payload = {
                    "text": text,
                    "model_id": "eleven_flash_v2_5", # low latency flash model
                    "voice_settings": {
                        "stability": 0.5,
                        "similarity_boost": 0.75
                    }
                }
                req = urllib.request.Request(
                    url,
                    data=json.dumps(payload).encode("utf-8"),
                    headers=headers,
                    method="POST"
                )
                with urllib.request.urlopen(req) as response:
                    audio_data = response.read()
                
                with open(temp_path, "wb") as wf:
                    wf.write(audio_data)
                
                # Play audio using mpv
                result = subprocess.run(
                    ["mpv", "--no-video", "--volume=100", temp_path],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                if result.returncode != 0:
                    raise RuntimeError(f"mpv exited with code {result.returncode}. Stderr: {result.stderr.decode()}")
            except Exception as e:
                import traceback
                err_msg = f"{e}"
                try:
                    with open("debug.log", "a") as df:
                        df.write(f"[{time.strftime('%H:%M:%S')}] ElevenLabs TTS Error: {err_msg}\n")
                        df.write(traceback.format_exc() + "\n")
                except Exception:
                    pass
                if error_callback:
                    loop.call_soon_threadsafe(error_callback, err_msg)
            finally:
                if os.path.exists(temp_path):
                    try:
                        os.unlink(temp_path)
                    except Exception:
                        pass
        await loop.run_in_executor(None, _speak)


    async def calibrate_noise(self, duration=1.5) -> float:
        """Calibrate background noise levels to establish VAD threshold."""
        loop = asyncio.get_running_loop()
        def _record():
            try:
                # Record silent room
                audio = sd.rec(int(duration * self.sample_rate), samplerate=self.sample_rate, channels=1, dtype='float32')
                sd.wait()
                rms = np.sqrt(np.mean(audio**2))
                return max(rms, 0.002)  # Ensure a minimum floor
            except Exception:
                return 0.01
        self.background_noise = await loop.run_in_executor(None, _record)
        return self.background_noise

    async def record_user_speech(self, level_callback=None, status_callback=None) -> np.ndarray:
        """Records microphone audio using dynamic VAD. Stops after sustained silence."""
        loop = asyncio.get_running_loop()
        
        def _record():
            threshold = self.background_noise * 2.5 + 0.005  # Dynamic threshold with safety margin
            block_size = 1024
            chunk_duration = block_size / self.sample_rate  # ~0.064s
            
            # Buffers
            pre_speech_buffer = []  # Store pre-speech chunks to avoid clipping start
            pre_speech_limit = int(0.3 / chunk_duration)  # Keep 0.3s of pre-speech
            speech_buffer = []
            
            state = "WAITING_FOR_SPEECH"
            silence_counter = 0
            silence_limit = int(self.silence_timeout / chunk_duration)
            consecutive_speech_counter = 0
            speech_trigger_limit = int(self.min_speech_duration / chunk_duration)
            max_chunks_limit = int(self.max_speech_duration / chunk_duration)
            
            if status_callback:
                loop.call_soon_threadsafe(status_callback, "Calibrated. Waiting for speech...")

            # Open stream
            try:
                with sd.InputStream(samplerate=self.sample_rate, channels=1, dtype='float32', blocksize=block_size) as stream:
                    total_chunks = 0
                    while total_chunks < max_chunks_limit:
                        data, overflowed = stream.read(block_size)
                        rms = np.sqrt(np.mean(data**2))
                        
                        if level_callback:
                            loop.call_soon_threadsafe(level_callback, rms)
                            
                        if state == "WAITING_FOR_SPEECH":
                            if rms > threshold:
                                consecutive_speech_counter += 1
                                pre_speech_buffer.append(data)
                                if len(pre_speech_buffer) > pre_speech_limit:
                                    pre_speech_buffer.pop(0)
                                    
                                if consecutive_speech_counter >= speech_trigger_limit:
                                    state = "RECORDING_SPEECH"
                                    if status_callback:
                                        loop.call_soon_threadsafe(status_callback, "Listening...")
                                    speech_buffer = list(pre_speech_buffer)
                            else:
                                consecutive_speech_counter = max(0, consecutive_speech_counter - 1)
                                pre_speech_buffer.append(data)
                                if len(pre_speech_buffer) > pre_speech_limit:
                                    pre_speech_buffer.pop(0)
                                    
                        elif state == "RECORDING_SPEECH":
                            speech_buffer.append(data)
                            total_chunks += 1
                            
                            if rms < threshold:
                                silence_counter += 1
                                if silence_counter >= silence_limit:
                                    if status_callback:
                                        loop.call_soon_threadsafe(status_callback, "Processing...")
                                    break
                            else:
                                silence_counter = 0
            except Exception as e:
                # If we fail to open microphone, log and return empty
                if status_callback:
                    loop.call_soon_threadsafe(status_callback, f"Audio error: {e}")
                return np.array([], dtype='float32')
                
            # If we timeout or stop, concatenate speech buffer
            if not speech_buffer:
                return np.array([], dtype='float32')
            return np.concatenate(speech_buffer, axis=0).flatten()
                
        return await loop.run_in_executor(None, _record)
