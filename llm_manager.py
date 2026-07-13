import asyncio
import os
import time
import json
from collections import deque
from groq import Groq

class LLMManager:
    def __init__(self, api_key: str, dispatch_manager, model_name="openai/gpt-oss-120b"):
        self.client = Groq(api_key=api_key)
        self.dispatch_manager = dispatch_manager
        self.model_name = model_name

        # TUI log for background communications
        self.agent_logs = []

        # Inboxes for background specialized agents only (NOT distributor)
        self.inboxes = {
            "police": [],
            "paramedics": [],
            "fire": []
        }

        # Structured FIFO task queue for the distributor.
        # Each item: {"from": str, "task": str}
        # Background agents and the dispatch system push tasks here.
        # The distributor picks ONE task per turn, executes it, then pops it.
        self.distributor_task_queue: deque = deque()

        # Track which dispatch keys we have already pushed arrival tasks for
        self._arrival_notified: set = set()

        # Conversation history for each agent in OpenAI format
        self.distributor_history = []
        self.police_history    = []
        self.paramedics_history = []
        self.fire_history      = []

        # Background agent activation queue
        self.agent_queue = asyncio.Queue()

        # Flag set by end_call tool to signal the main loop to terminate
        self.call_ended = False

        # Start background agent orchestrator
        self.worker_task = asyncio.create_task(self.start_background_worker())

    # ------------------------------------------------------------------ #
    # Background worker                                                    #
    # ------------------------------------------------------------------ #

    async def start_background_worker(self):
        """Asynchronously processes background agent communications."""
        while True:
            agent_name = await self.agent_queue.get()
            try:
                await self.run_agent(agent_name)
            except Exception as e:
                import traceback
                self.agent_logs.append(f"[System Error]: Failed running {agent_name} agent: {e}")
                with open("debug.log", "a") as df:
                    df.write(f"Background agent error for {agent_name}: {e}\n")
                    df.write(traceback.format_exc() + "\n")
            finally:
                self.agent_queue.task_done()

    # ------------------------------------------------------------------ #
    # Arrival polling — called at the top of every process_turn           #
    # ------------------------------------------------------------------ #

    def _check_arrivals(self):
        """Push arrival notification tasks for any newly-arrived units."""
        for key, dispatch in self.dispatch_manager.dispatches.items():
            if dispatch.status == "Arrived" and key not in self._arrival_notified:
                self._arrival_notified.add(key)
                task = (
                    f"ARRIVAL ALERT: {dispatch.service_type} units have just ARRIVED at "
                    f"'{dispatch.location}'. Immediately inform the caller that "
                    f"{dispatch.service_type} is on scene. Ask them to confirm they can see the "
                    f"responders. Once confirmed, use the end_call tool to close the call."
                )
                self.distributor_task_queue.appendleft(  # highest priority — goes to front
                    {"from": "system", "task": task}
                )
                self.agent_logs.append(
                    f"[System ➔ Distributor]: Arrival task pushed for {dispatch.service_type}."
                )

    # ------------------------------------------------------------------ #
    # Tool schemas & local functions                                       #
    # ------------------------------------------------------------------ #

    def _make_send_message_for(self, sender_name: str):
        """
        Returns (schema, fn) for the send_message tool bound to a specific sender.
        When recipient is 'distributor', the message is pushed as a structured task.
        When recipient is another agent, it goes to that agent's inbox.
        """
        schema = {
            "type": "function",
            "function": {
                "name": "send_message",
                "description": (
                    "Send a message to another agent. "
                    "When sending to 'distributor', write a SINGLE concise question or notification "
                    "(one sentence only) that the distributor should relay to the caller."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "recipient": {
                            "type": "string",
                            "enum": ["police", "paramedics", "fire", "distributor"],
                            "description": "The target agent."
                        },
                        "message": {
                            "type": "string",
                            "description": (
                                "The message. If recipient is 'distributor', this must be "
                                "ONE single-sentence question or update for the caller."
                            )
                        }
                    },
                    "required": ["recipient", "message"]
                }
            }
        }

        def send_message(recipient: str, message: str) -> str:
            log_entry = f"[{sender_name.capitalize()} ➔ {recipient.capitalize()}]: {message}"
            self.agent_logs.append(log_entry)

            if recipient == "distributor":
                # Push as a structured task onto the distributor's queue
                self.distributor_task_queue.append(
                    {"from": sender_name, "task": message}
                )
            elif recipient in self.inboxes:
                self.inboxes[recipient].append(f"From {sender_name}: {message}")
                self.agent_queue.put_nowait(recipient)
            else:
                return f"Error: unknown recipient '{recipient}'."

            return f"Message delivered to {recipient}."

        return schema, send_message

    def _get_agent_tools_and_funcs(self, name: str):
        """Returns tool schemas and local function map for a given agent."""
        send_message_schema, send_message_fn = self._make_send_message_for(name)

        # Distributor-only tools
        if name == "distributor":
            end_call_schema = {
                "type": "function",
                "function": {
                    "name": "end_call",
                    "description": (
                        "Terminate the 911 call. Call ONLY after the caller has confirmed "
                        "that emergency responders have arrived and assistance is underway."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "reason": {
                                "type": "string",
                                "description": "Brief reason (e.g. 'Caller confirmed paramedics on scene')."
                            }
                        },
                        "required": ["reason"]
                    }
                }
            }

            def end_call(reason: str = "") -> str:
                self.call_ended = True
                self.agent_logs.append(f"[System]: Call ended — {reason}")
                return "Call terminated."

            return (
                [send_message_schema, end_call_schema],
                {"send_message": send_message_fn, "end_call": end_call}
            )

        # Specialized agent tools
        if name == "police":
            dispatch_schema = {
                "type": "function",
                "function": {
                    "name": "dispatch_police",
                    "description": "Dispatch police units to a location.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {"type": "string"},
                            "details":  {"type": "string"},
                            "priority": {"type": "string", "enum": ["low", "medium", "high"]}
                        },
                        "required": ["location", "details"]
                    }
                }
            }
            def dispatch_police(location: str, details: str, priority: str = "medium") -> str:
                eta = 3 if priority == "high" else (5 if priority == "medium" else 8)
                self.dispatch_manager.dispatch("police", location, f"[{priority.upper()}] {details}", eta)
                self.agent_logs.append(f"[Police ➔ System]: Dispatched to {location} (ETA {eta} min).")
                return f"Police dispatched to {location}. Priority: {priority}. ETA: {eta} min."

            return (
                [send_message_schema, dispatch_schema],
                {"send_message": send_message_fn, "dispatch_police": dispatch_police}
            )

        elif name == "paramedics":
            dispatch_schema = {
                "type": "function",
                "function": {
                    "name": "dispatch_paramedics",
                    "description": "Dispatch paramedic/ambulance units to a location.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {"type": "string"},
                            "details":  {"type": "string"},
                            "severity": {"type": "string", "enum": ["minor", "moderate", "critical"]}
                        },
                        "required": ["location", "details"]
                    }
                }
            }
            def dispatch_paramedics(location: str, details: str, severity: str = "moderate") -> str:
                eta = 4 if severity == "critical" else (6 if severity == "moderate" else 10)
                self.dispatch_manager.dispatch("paramedics", location, f"[{severity.upper()}] {details}", eta)
                self.agent_logs.append(f"[Paramedics ➔ System]: Dispatched to {location} (ETA {eta} min).")
                return f"Paramedics dispatched to {location}. Severity: {severity}. ETA: {eta} min."

            return (
                [send_message_schema, dispatch_schema],
                {"send_message": send_message_fn, "dispatch_paramedics": dispatch_paramedics}
            )

        elif name == "fire":
            dispatch_schema = {
                "type": "function",
                "function": {
                    "name": "dispatch_fire",
                    "description": "Dispatch fire engines or rescue teams to a location.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location":     {"type": "string"},
                            "details":      {"type": "string"},
                            "hazard_level": {"type": "string", "enum": ["low", "moderate", "severe"]}
                        },
                        "required": ["location", "details"]
                    }
                }
            }
            def dispatch_fire(location: str, details: str, hazard_level: str = "moderate") -> str:
                eta = 4 if hazard_level == "severe" else (6 if hazard_level == "moderate" else 9)
                self.dispatch_manager.dispatch("fire", location, f"[{hazard_level.upper()}] {details}", eta)
                self.agent_logs.append(f"[Fire ➔ System]: Dispatched to {location} (ETA {eta} min).")
                return f"Fire dispatched to {location}. Hazard: {hazard_level}. ETA: {eta} min."

            return (
                [send_message_schema, dispatch_schema],
                {"send_message": send_message_fn, "dispatch_fire": dispatch_fire}
            )

        return [send_message_schema], {"send_message": send_message_fn}

    # ------------------------------------------------------------------ #
    # System instructions                                                  #
    # ------------------------------------------------------------------ #

    def _get_agent_system_instruction(self, name: str) -> str:
        if name == "police":
            return (
                "You are the 911 Police Dispatch Agent.\n"
                "On receiving a request: use dispatch_police immediately. Then send the distributor "
                "EXACTLY TWO separate send_message calls — each containing ONE single-sentence question "
                "critical for the responding officers (e.g. weapon type, suspect description, "
                "number of suspects). Do not send lists. One question per send_message call. "
                "Do not speak to the caller directly."
            )
        elif name == "paramedics":
            return (
                "You are the 911 Paramedics Dispatch Agent.\n"
                "On receiving a request: use dispatch_paramedics immediately. Then send the distributor "
                "EXACTLY TWO separate send_message calls — each containing ONE single-sentence question "
                "critical for the responding paramedics (e.g. consciousness, known medications, "
                "allergies, injury type). Do not send lists. One question per send_message call. "
                "Do not speak to the caller directly."
            )
        elif name == "fire":
            return (
                "You are the 911 Fire & Rescue Dispatch Agent.\n"
                "On receiving a request: use dispatch_fire immediately. Then send the distributor "
                "EXACTLY TWO separate send_message calls — each containing ONE single-sentence question "
                "critical for the responding firefighters (e.g. trapped occupants, gas lines, "
                "building materials, hazardous materials). Do not send lists. One question per "
                "send_message call. Do not speak to the caller directly."
            )
        return ""

    def _get_distributor_system_instruction(self, current_task: dict | None) -> str:
        dispatches = self.dispatch_manager.get_active_dispatches_str()

        pending_count = len(self.distributor_task_queue)
        pending_note = (
            f"({pending_count - 1} more task(s) waiting after this one)"
            if pending_count > 1 else ""
        )

        if current_task:
            task_block = (
                f"━━━ MANDATORY CURRENT TASK {pending_note} ━━━\n"
                f"From: {current_task['from'].upper()}\n"
                f"Task: {current_task['task']}\n"
                f"You MUST address this task in your response before anything else. "
                f"If it is a question, ask the caller right now. "
                f"If it is an arrival alert, announce it and ask the caller to confirm.\n"
                f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
            )
        else:
            task_block = "No pending tasks. Continue engaging and reassuring the caller."

        return f"""You are the primary 911 distributor operator speaking directly to the caller.

RULES (follow strictly):
1. Establish the caller's location and nature of emergency first.
2. Delegate emergencies using send_message:
   - Crime / safety → 'police'
   - Medical → 'paramedics'
   - Fire / hazard → 'fire'
   You have NO dispatch tools. Only specialized agents can dispatch.
3. Keep the caller calm. Give concise safety instructions.
4. When the caller answers a question from a background agent, relay that answer back to the relevant agent via send_message.
5. If caller goes silent, reassure and keep the call alive.
6. After the caller confirms responders are on scene, call end_call immediately.

FORMATTING (STRICT — violations break TTS):
- Output ONLY raw spoken dialogue. NO speaker labels, NO 'Operator:', NO markdown symbols (**, *, #, -, `), NO bullet points, NO numbered lists.
- 2 to 3 natural conversational sentences per response max.

{task_block}

Current Active Dispatches:
{dispatches}
"""

    # ------------------------------------------------------------------ #
    # Run a background specialized agent                                   #
    # ------------------------------------------------------------------ #

    async def run_agent(self, name: str):
        inbox = self.inboxes.get(name, [])
        if not inbox:
            return

        inbox_msgs = list(inbox)
        inbox.clear()

        dispatches = self.dispatch_manager.get_active_dispatches_str()
        system_instruction = self._get_agent_system_instruction(name)

        prompt = (
            f"Active dispatches:\n{dispatches}\n\n"
            f"Your inbox:\n" + "\n".join(f"- {m}" for m in inbox_msgs) + "\n\n"
            "Take action now."
        )

        history_list: list = getattr(self, f"{name}_history")
        history_list.append({"role": "user", "content": prompt})

        tools, local_functions = self._get_agent_tools_and_funcs(name)
        loop = asyncio.get_running_loop()

        while True:
            messages = [{"role": "system", "content": system_instruction}] + history_list

            response = await loop.run_in_executor(None, lambda: self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                tools=tools or None,
                tool_choice="auto" if tools else None,
                temperature=0.2
            ))

            response_message = response.choices[0].message

            if response_message.tool_calls:
                history_list.append(response_message.model_dump(exclude_none=True))
                for tool_call in response_message.tool_calls:
                    fn_name = tool_call.function.name
                    fn_args = json.loads(tool_call.function.arguments)
                    try:
                        result = local_functions[fn_name](**fn_args) if fn_name in local_functions else f"Unknown tool: {fn_name}"
                    except Exception as e:
                        result = f"Error in {fn_name}: {e}"
                    history_list.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": fn_name,
                        "content": result
                    })
            else:
                history_list.append(response_message.model_dump(exclude_none=True))
                break

    # ------------------------------------------------------------------ #
    # Main distributor turn                                                #
    # ------------------------------------------------------------------ #

    async def process_turn(self, user_text: str, on_tool_call_callback=None) -> tuple[str, bool]:
        """Process one caller turn through the distributor agent.

        Returns:
            (response_text, call_ended)
        """
        self.call_ended = False

        # Check for newly-arrived units and push arrival tasks
        self._check_arrivals()

        # Peek at the next pending task (do NOT pop yet — pop after the turn)
        current_task = self.distributor_task_queue[0] if self.distributor_task_queue else None

        self.distributor_history.append({"role": "user", "content": user_text})

        tools, local_functions = self._get_agent_tools_and_funcs("distributor")
        loop = asyncio.get_running_loop()

        while True:
            system_instruction = self._get_distributor_system_instruction(current_task)
            messages = [{"role": "system", "content": system_instruction}] + self.distributor_history

            response = await loop.run_in_executor(None, lambda: self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                tools=tools or None,
                tool_choice="auto" if tools else None,
                temperature=0.3
            ))

            response_message = response.choices[0].message

            if response_message.tool_calls:
                self.distributor_history.append(response_message.model_dump(exclude_none=True))

                for tool_call in response_message.tool_calls:
                    fn_name = tool_call.function.name
                    fn_args = json.loads(tool_call.function.arguments)

                    if on_tool_call_callback:
                        on_tool_call_callback(fn_name, fn_args)

                    try:
                        result = local_functions[fn_name](**fn_args) if fn_name in local_functions else f"Unknown tool: {fn_name}"
                    except Exception as e:
                        result = f"Error in {fn_name}: {e}"

                    self.distributor_history.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": fn_name,
                        "content": result
                    })
            else:
                self.distributor_history.append(response_message.model_dump(exclude_none=True))

                # Pop the completed task now that the turn is done
                if current_task and self.distributor_task_queue and self.distributor_task_queue[0] is current_task:
                    self.distributor_task_queue.popleft()

                return response_message.content or "", self.call_ended
