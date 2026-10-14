import asyncio
import os
import re
import sys
from dotenv import load_dotenv

def sanitize_response(text: str) -> str:
    """Strips markdown symbols and speaker-label prefixes from LLM responses for clean TTS."""
    # Remove bold/italic markers
    text = re.sub(r'\*+', '', text)
    # Remove heading markers
    text = re.sub(r'^#+\s*', '', text, flags=re.MULTILINE)
    # Remove backticks
    text = re.sub(r'`+', '', text)
    # Remove leading bullet / dash list markers
    text = re.sub(r'^\s*[-•]\s+', '', text, flags=re.MULTILINE)
    # Remove speaker-label prefixes like "Operator:", "Operator Response:"
    text = re.sub(r'^(?:Operator|Dispatcher|Operator Response)\s*:\s*', '', text, flags=re.IGNORECASE | re.MULTILINE)
    # Collapse multiple blank lines
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

# Load environment variables from .env if present
load_dotenv()

# Check for API Keys first to prevent mid-TUI crashes
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    print("\n🚨 GROQ_API_KEY not found in environment or .env file.")
    groq_api_key = input("🔑 Please enter your Groq API Key: ").strip()
    if not groq_api_key:
        print("❌ Groq API key required. Exiting.")
        sys.exit(1)

prod_env = os.getenv("PROD", "true").strip().lower() != "false"
elevenlabs_api_key = os.getenv("ELEVENLABS_API_KEY")
if not elevenlabs_api_key and prod_env:
    print("\n🚨 ELEVENLABS_API_KEY not found in environment or .env file.")
    elevenlabs_api_key = input("🔑 Please enter your ElevenLabs API Key: ").strip()
    if not elevenlabs_api_key:
        print("❌ ElevenLabs API key required. Exiting.")
        sys.exit(1)

from rich.console import Console
from rich.live import Live
from rich.text import Text
from rich.table import Table

import select
import tty
import termios
import time
import numpy as np

# Import our custom modules
from audio_manager import AudioManager
from speech_analyzer import SpeechAnalyzer
from dispatch_manager import DispatchManager
from llm_manager import LLMManager

class TerminalInputManager:
    """Manages non-blocking keyboard input for Push-to-Talk and interruption in terminal."""
    def __init__(self):
        self.old_settings = None
        self.is_raw = False

    def enable_cbreak(self):
        try:
            if sys.stdin.isatty():
                self.old_settings = termios.tcgetattr(sys.stdin)
                tty.setcbreak(sys.stdin.fileno())
                self.is_raw = True
        except Exception:
            pass

    def restore(self):
        try:
            if self.is_raw and self.old_settings and sys.stdin.isatty():
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_settings)
                self.is_raw = False
        except Exception:
            pass

    def check_key(self) -> str | None:
        try:
            if sys.stdin.isatty() and select.select([sys.stdin], [], [], 0)[0]:
                return sys.stdin.read(1)
        except Exception:
            pass
        return None

class Dashboard:
    def __init__(self, dispatch_manager):
        self.dispatch_manager = dispatch_manager
        self.llm = None
        self.transcript = []  # list of (speaker, text) tuples
        self.status = "Initializing..."
        self.audio_level = 0.0
        self.layout = self._create_layout()

    def _create_layout(self) -> Table:
        # We will use a main grid Table to represent the columns for robust terminal compatibility
        # instead of the full rich.Layout, as tables are easier to render reliably inside Live.
        table = Table.grid(expand=True)
        table.add_column("left_col", ratio=3)
        table.add_column("spacer", width=2)
        table.add_column("right_col", ratio=2)
        return table

    def add_transcript(self, speaker: str, text: str):
        self.transcript.append((speaker, text))
        # Keep last 10 dialogs to fit on standard terminal sizes
        if len(self.transcript) > 10:
            self.transcript.pop(0)

    def set_status(self, status: str):
        self.status = status

    def set_audio_level(self, level: float):
        self.audio_level = level

    def get_header(self) -> Table:
        header_grid = Table.grid(expand=True)
        header_grid.add_column(justify="center")
        header_grid.add_row(Text("🚨 911 EMERGENCY DISPATCH SYSTEM 🚨", style="bold red blink"))
        header_grid.add_row(Text("─" * 60, style="dim red"))
        return header_grid

    def get_transcript_panel(self) -> Table:
        panel_table = Table(title="📞 CALL DIALOGUE LOG", expand=True, border_style="green")
        panel_table.add_column("Speaker", width=12)
        panel_table.add_column("Dialogue")
        
        for speaker, msg in self.transcript:
            if speaker == "Caller":
                panel_table.add_row(Text("Caller", style="bold green"), Text(msg, style="white"))
            elif speaker == "Operator":
                panel_table.add_row(Text("Operator", style="bold yellow"), Text(msg, style="cyan"))
            elif speaker == "System":
                panel_table.add_row(Text("System", style="bold dim white"), Text(msg, style="dim white"))
        return panel_table

    def get_dispatches_panel(self) -> Table:
        panel_table = Table(title="🚑 ACTIVE DISPATCHES", expand=True, border_style="yellow")
        panel_table.add_column("Service", style="bold")
        panel_table.add_column("Status", style="bold")
        panel_table.add_column("ETA", justify="right")
        panel_table.add_column("Location & Details", style="dim")

        for key, item in self.dispatch_manager.dispatches.items():
            status_style = "bold red"
            if item.status == "En Route":
                status_style = "bold yellow"
            elif item.status == "Arrived":
                status_style = "bold green"

            eta_str = f"{item.eta}m" if item.eta > 0 else "ARRIVED"
            
            emoji = "police"
            if "paramedic" in key or "medical" in key or "ambulance" in key:
                emoji = "🚑 medical"
            elif "fire" in key:
                emoji = "🚒 fire"
            elif "police" in key:
                emoji = "🚓 police"
            else:
                emoji = f"🚨 {item.service_type}"
                
            panel_table.add_row(
                emoji,
                Text(item.status, style=status_style),
                Text(eta_str, style="bold cyan"),
                f"{item.location} - {item.details}"
            )
        return panel_table

    def get_agent_logs_panel(self) -> Table:
        panel_table = Table(title="💬 AGENT COLLABORATION LOG", expand=True, border_style="blue")
        panel_table.add_column("Agent Communication", style="cyan")
        
        logs = self.llm.agent_logs if self.llm else []
        for log in logs[-8:]:
            # Highlight messages using style formatting based on sender/receiver
            if "[Police" in log:
                color = "blue"
            elif "[Paramedics" in log:
                color = "magenta"
            elif "[Fire" in log:
                color = "red"
            else:
                color = "yellow"
            panel_table.add_row(Text(log, style=color))
        return panel_table

    def get_footer(self) -> Table:
        footer_grid = Table.grid(expand=True)
        footer_grid.add_column(ratio=3)
        footer_grid.add_column(ratio=2)
        
        # Audio bar
        bar_width = 25
        # RMS is typically small, let's scale it so normal speech fills it nicely
        filled = min(bar_width, int(self.audio_level * bar_width * 15))
        bar = "█" * filled + "░" * (bar_width - filled)
        
        audio_color = "green"
        if self.audio_level > 0.08:
            audio_color = "red"
        elif self.audio_level > 0.03:
            audio_color = "yellow"
            
        audio_bar_text = Text(f"Mic Level: [{bar}]", style=f"bold {audio_color}")
        status_text = Text(f"Status: {self.status}\n[SPACE]: Push-to-Talk / Interrupt Operator | [q]: Quit", style="bold white")
        
        footer_grid.add_row(status_text, audio_bar_text)
        
        outer_grid = Table.grid(expand=True)
        outer_grid.add_row(Text("─" * 60, style="dim blue"))
        outer_grid.add_row(footer_grid)
        return outer_grid

    def render(self) -> Table:
        grid = Table.grid(expand=True)
        grid.add_row(self.get_header())
        
        body_grid = Table.grid(expand=True)
        body_grid.add_column("left", ratio=3)
        body_grid.add_column("spacer", width=2)
        body_grid.add_column("right", ratio=2)
        
        # Create a right-hand stack table
        right_stack = Table.grid(expand=True)
        right_stack.add_row(self.get_dispatches_panel())
        right_stack.add_row(Text(""))  # spacer
        right_stack.add_row(self.get_agent_logs_panel())
        
        body_grid.add_row(self.get_transcript_panel(), "", right_stack)
        
        grid.add_row(body_grid)
        grid.add_row(self.get_footer())
        return grid


async def main():
    console = Console()
    dispatch_manager = DispatchManager()
    dashboard = Dashboard(dispatch_manager)

    # Start Rich Live TUI
    with Live(dashboard.render(), console=console, refresh_per_second=10) as live:
        # Update TUI loop
        async def tui_refresh():
            while True:
                live.update(dashboard.render())
                await asyncio.sleep(0.1)

        asyncio.create_task(tui_refresh())
        
        # Start Dispatch Background Simulation Tick
        # ETA ticks down 1 simulated minute every 10 seconds of real-world time
        asyncio.create_task(dispatch_manager.start_simulation_loop(on_update_callback=None))

        # Start Loader Steps
        dashboard.set_status("Initializing Speech-to-Text Session...")
        dashboard.add_transcript("System", "Connecting to Groq Cloud STT...")
        
        try:
            # Load Speech Analyzer (now using Groq Whisper API)
            analyzer = SpeechAnalyzer(api_key=groq_api_key)
            dashboard.add_transcript("System", "Groq STT online.")
        except Exception as e:
            dashboard.add_transcript("System", f"STT Initialization failed: {e}")
            await asyncio.sleep(3)
            return

        dashboard.set_status("Initializing Audio Stream...")
        audio = AudioManager(api_key=elevenlabs_api_key, voice_name="pNInz6obpgDQGcFmaJgB", silence_timeout=1.5)
        
        dashboard.set_status("Calibrating background noise (please remain silent)...")
        background_noise = await audio.calibrate_noise(duration=1.5)
        dashboard.add_transcript("System", f"Microphone calibrated. Noise floor: {background_noise:.5f}")

        # Initialize LLM Manager
        dashboard.set_status("Initializing Operator Session...")
        llm = LLMManager(api_key=groq_api_key, dispatch_manager=dispatch_manager)
        dashboard.llm = llm
        dashboard.add_transcript("System", "911 Operator System online. Connecting incoming call...")
        await asyncio.sleep(1.5)

        # Define TTS error logger
        def on_tts_error(err_msg):
            dashboard.add_transcript("System", f"⚠️ TTS Error: {err_msg}")

        # 911 Greeting
        greeting = "911, what is the location of your emergency?"
        dashboard.add_transcript("Operator", greeting)
        dashboard.set_status("Operator Speaking...")
        await audio.speak(greeting, error_callback=on_tts_error)

        input_mgr = TerminalInputManager()
        input_mgr.enable_cbreak()

        silence_timer_start = time.time()
        silence_state = "NORMAL"  # "NORMAL", "CHECK_IN", "MONITORING"
        has_caller_spoken_yet = False
        SILENCE_LIMIT_NORMAL = 15.0
        SILENCE_LIMIT_CHECK_IN = 5.0
        SILENCE_LIMIT_MONITORING = 25.0

        try:
            # Call Interaction Loop
            while True:
                if silence_state == "CHECK_IN":
                    current_limit = SILENCE_LIMIT_CHECK_IN
                    dashboard.set_status("Awaiting check-in response (Press SPACE to speak)...")
                elif silence_state == "MONITORING":
                    current_limit = SILENCE_LIMIT_MONITORING
                    dashboard.set_status("Monitoring line (Press SPACE to speak / interrupt)...")
                else:
                    current_limit = SILENCE_LIMIT_NORMAL
                    dashboard.set_status("Awaiting caller (Press SPACE to speak / interrupt)...")

                user_triggered_speech = False
                user_audio = np.array([], dtype='float32')

                # Inner polling loop: checks keypresses and countdown timer
                while True:
                    if audio.is_speaking():
                        silence_timer_start = time.time()

                    elapsed_silence = time.time() - silence_timer_start if not audio.is_speaking() else 0.0

                    k = input_mgr.check_key()
                    if k == 'q':
                        dashboard.set_status("Call Ended by User")
                        audio.stop_speaking()
                        return
                    elif k in (' ', '\n', '\r'):
                        # Instantly stop TTS if operator is speaking
                        if audio.is_speaking():
                            audio.stop_speaking()

                        # Record PTT audio
                        dashboard.set_status("🎙️ Recording Speech (Press SPACE/ENTER when done)...")
                        user_audio = await asyncio.to_thread(
                            audio.record_ptt_speech,
                            level_callback=dashboard.set_audio_level,
                            status_callback=dashboard.set_status,
                            key_check_fn=input_mgr.check_key
                        )
                        dashboard.set_audio_level(0.0)
                        user_triggered_speech = True
                        break

                    if not audio.is_speaking() and elapsed_silence >= current_limit:
                        break

                    await asyncio.sleep(0.05)

                interrupted_text = audio.get_and_clear_interrupted_text()
                transcription = ""

                if user_triggered_speech and len(user_audio) > 0:
                    dashboard.set_status("Transcribing speech...")
                    transcription = await analyzer.transcribe(user_audio)

                is_valid_speech = bool(
                    transcription
                    and transcription.strip() != ""
                    and not transcription.startswith("[Groq STT failed")
                    and not transcription.startswith("[Caller is silent")
                )

                if is_valid_speech:
                    dashboard.add_transcript("Caller", transcription)
                    has_caller_spoken_yet = True
                    silence_state = "NORMAL"
                    silence_timer_start = time.time()
                    prompt_input = transcription
                else:
                    if silence_state == "NORMAL":
                        # Stage 1 Timeout reached -> Trigger check-in
                        silence_state = "CHECK_IN"
                        dashboard.add_transcript("System", "⏰ Caller silent for 15s. Triggering check-in...")
                        if not has_caller_spoken_yet:
                            prompt_input = "[SYSTEM ALERT: Open-line silence. The caller has been silent for 15 seconds without speaking or providing a location. Ask a short check-in question instructing them to speak, press space, or tap if they need emergency assistance.]"
                        else:
                            prompt_input = "[SYSTEM ALERT: Mid-call silence. The caller has stopped speaking for 15 seconds. Ask a short check-in question to confirm if they are still on the line.]"
                    elif silence_state == "CHECK_IN":
                        # Stage 2 Timeout reached -> Transition to MONITORING
                        silence_state = "MONITORING"
                        if not has_caller_spoken_yet:
                            dashboard.add_transcript("System", "📞 Open-line unverified. Holding line and requesting location...")
                            prompt_input = "[SYSTEM ALERT: Open-line unverified. The caller failed to respond to the check-in and no location or emergency type has been provided. Tell the caller you are holding the line, explain that you need their location/address to send help, and instruct them to press space or tap if in danger. Do NOT dispatch units or say units are en route.]"
                        else:
                            dashboard.add_transcript("System", "🚨 Mid-call silence: Auto-escalating with emergency update...")
                            prompt_input = "[CRITICAL ALERT: Mid-call unresponsiveness. The caller failed to respond to check-in. Based on the emergency context established earlier, deduce the emergency (e.g. lost consciousness/collapsed), alert the specialized agent via send_message, and reassure the caller that help is on the way.]"
                    elif silence_state == "MONITORING":
                        # In monitoring mode, send brief periodic reassurance without looping into the check-in
                        if not has_caller_spoken_yet:
                            prompt_input = "[SYSTEM ALERT: Open-line holding. Reassure the caller you are still on the line waiting for their location. Keep it to one short sentence. Do not repeat long questions.]"
                        else:
                            prompt_input = "[SYSTEM ALERT: Periodic reassurance. Reassure the caller in one short sentence that emergency units are en route and to stay calm.]"
                    else:
                        prompt_input = "[Caller is silent or unresponsive]"

                # 3. Think / Generate Response
                dashboard.set_status("Operator thinking...")

                def on_tool_call(name, args):
                    service = args.get("service_type", "emergency unit")
                    loc = args.get("location", "specified location")
                    details = args.get("details", "")
                    details_str = f" ({details})" if details else ""
                    dashboard.add_transcript("System", f"📢 Tool Invoked: Dispatched {service} to {loc}{details_str}")

                try:
                    response_text, call_ended = await llm.process_turn(
                        prompt_input,
                        on_tool_call_callback=on_tool_call,
                        interrupted_text=interrupted_text
                    )
                except Exception as e:
                    response_text, call_ended = "We are coordinating emergency responders to your location. Please stay on the line.", False
                    dashboard.add_transcript("System", f"LLM API Error: {e}")

                # 4. Sanitize and speak response
                response_text = sanitize_response(response_text)
                dashboard.add_transcript("Operator", response_text)
                dashboard.set_status("Operator Speaking...")
                asyncio.create_task(audio.speak(response_text, error_callback=on_tts_error))
                silence_timer_start = time.time()

                # 5. End call if responders confirmed
                if call_ended:
                    dashboard.set_status("Call Ended")
                    dashboard.add_transcript("System", "Call terminated. Responders on scene.")
                    audio.stop_speaking()
                    break
        finally:
            input_mgr.restore()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 911 Dispatch Session ended by operator.")
        sys.exit(0)
