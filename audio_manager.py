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
        self.active_playback_process = None
        self._last_transcription = ""

    def is_speaking(self) -> bool:
        if self.active_playback_process is not None:
            if self.active_playback_process.poll() is None:
                return True
            else:
                self.active_playback_process = None
        return False

    def stop_speaking(self):
        if self.active_playback_process is not None:
            try:
                self.active_playback_process.terminate()
                try:
                    self.active_playback_process.wait(timeout=0.5)
                except subprocess.TimeoutExpired:
                    self.active_playback_process.kill()
            except Exception:
                pass
            finally:
                self.active_playback_process = None

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
                self.active_playback_process = subprocess.Popen(
                    ["mpv", "--no-video", "--volume=100", temp_path],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                stdout, stderr = self.active_playback_process.communicate()
                returncode = self.active_playback_process.returncode
                if returncode != 0 and returncode != -15 and returncode != -9:
                    raise RuntimeError(f"mpv exited with code {returncode}. Stderr: {stderr.decode()}")
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
                self.active_playback_process = None
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

    def _record_loop(self, level_callback, status_callback, loop) -> np.ndarray:
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
            msg = "Operator Speaking (Listening...)" if self.is_speaking() else "Calibrated. Waiting for speech..."
            loop.call_soon_threadsafe(status_callback, msg)

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

    async def record_user_speech(self, level_callback=None, status_callback=None, transcribe_callback=None) -> np.ndarray:
        """Records microphone audio using dynamic VAD. Stops after sustained silence.
        If transcribe_callback is provided, it transcribes the recorded speech.
        If the transcription is non-empty, it stops any active operator speech.
        If the transcription is empty and the operator is still speaking, it loops to continue listening.
        """
        loop = asyncio.get_running_loop()
        
        while True:
            user_audio = await loop.run_in_executor(None, self._record_loop, level_callback, status_callback, loop)
            
            if len(user_audio) == 0 or transcribe_callback is None:
                return user_audio
                
            # Transcribe the audio chunk
            transcription = await transcribe_callback(user_audio)
            
            # If transcription is valid (not empty/noise)
            if transcription and transcription.strip() != "" and not transcription.startswith("[Groq STT failed") and not transcription.startswith("[Caller is silent"):
                # We got real speech! Stop the operator if still speaking
                self.stop_speaking()
                # Store the transcription so we don't have to transcribe it again
                self._last_transcription = transcription
                return user_audio
            
            # If transcription is empty/noise:
            # Check if operator is still speaking
            if self.is_speaking():
                # Operator is still speaking! The trigger was just noise.
                # Loop again to keep listening.
                if status_callback:
                    loop.call_soon_threadsafe(status_callback, "Operator Speaking...")
                continue
            else:
                # Operator has finished, and caller was silent or just made noise.
                # Clear last transcription and return empty.
                self._last_transcription = ""
                return np.array([], dtype='float32')
