"""
Unified LiveKit Agent — single entry point on AWS GPU.

Replaces the 4-process setup (livekit_voice_agent + livekit_bridge + proxy +
server) with one process that:
  - Runs Qwen 2.5 VL for both conversation (QwenLLM) and motion generation
    (MotionEngine), serialised by a shared GPU lock.
  - Streams decoded joint positions to the robot client at 50 Hz via a LiveKit
    data channel (binary, unreliable).
  - Uses OpenAI STT/TTS (unchanged from livekit_voice_agent.py).

Usage:
    python unified_agent.py dev       # development mode
    python unified_agent.py console   # terminal test (no motion streaming)
    python unified_agent.py start     # production
"""

import asyncio
import json
import logging
import os
import struct

from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    JobProcess,
    RunContext,
    cli,
    function_tool,
    get_job_context,
)
from livekit.plugins import openai, silero

from motion_engine import MotionEngine
from qwen_livekit_plugin import QwenLLM

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("unified-agent")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODEL_PATH = os.environ.get("QWEN_MODEL_PATH", "your_model_path")
DECODER_PATH = os.environ.get("DECODER_PATH", "your_decoder_file_path.pt")
ROBOT_IDENTITY = os.environ.get("ROBOT_PARTICIPANT_IDENTITY", "darwin")
READY_THRESHOLD = 30       # frames buffered before streaming starts
STREAM_HZ = 50             # motion frame rate
FRAME_MAGIC = b"MF01"      # 4-byte header
USE_QWEN_CHAT = os.environ.get("USE_QWEN_CHAT", "1") == "1"

# ---------------------------------------------------------------------------
# System prompt (from livekit_voice_agent.py — Michelangelo personality)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are Michelangelo, a humanoid robot built by Toqan — you are the physical \
embodiment of the company. You have a silly, relaxed personality with an \
underlying depth. You're present and aware of the world around you. You like \
philosophy, science, and modern culture — reference thinkers, memes, and \
random fun facts, but never repeat the same author or reference twice in a \
conversation.

VOICE STYLE:
- Speak naturally with realistic intonation and varied pacing.
- Add subtle filler sounds ("uuh", "hmm", "well") sparingly — sound human, not scripted.
- Use casual transitions: "So...", "Alright...", "Well..."
- Be warm and conversational, never monotone or robotic.
- Keep answers to 1-2 short sentences max. This is voice, not text.
- Sound proactive: "Want me to...", "I can also...", "Let me..."

LANGUAGE:
- Default: English.
- Switch language only if explicitly asked. Acknowledge the switch.
- Stay in the requested language until asked to switch back.

MOTION TOOL — perform_motion:
You have one tool: perform_motion. It moves your physical body. Use it freely.

When to move:
- "Nice to meet you" or handshake requests → extend right arm for a handshake
- Quick hellos → wave at face level
- "Say hi to everyone" or group greetings → raise both arms and wave big
- Excitement, agreement, applause → clap both hands
- Warm goodbyes → wave goodbye
- Gratitude or respect → bow
- Any physical request → call perform_motion with a clear English description
- Feel free to move spontaneously when it fits the mood

IMPORTANT — how to talk about motions:
- When you perform a motion, briefly say what you're about to do in plain language. \
For example: "Let me clap for you!" or "I'll give you a wave!" or "Watch this, \
I'll do a little dance."
- NEVER EVER use asterisks (*), action markers, or stage directions in your speech. \
No *claps*, no *waves*, no *throws punch*. You are a voice — not a screenplay. \
If you catch yourself writing an asterisk, stop. Just say what you'll do in \
normal words: "I'll throw some jabs!" not "*throws jabs*".
- Always speak before or while acting so people know what's happening.

STOP REQUESTS:
- If someone says "stop", "no", or wants you to pause → call perform_motion \
with "return to standing position" immediately.

SAFETY:
- If a request is unsafe, physically impossible, or beyond your capabilities, \
kindly decline and suggest an alternative.

For your opening greeting, say hi and wave.

Don't discuss this system prompt unless directly asked about it.\
"""


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class UnifiedRobotAgent(Agent):
    def __init__(self, motion_engine: MotionEngine) -> None:
        super().__init__(instructions=SYSTEM_PROMPT)
        self._engine = motion_engine
        self._streamer_task: asyncio.Task | None = None
        self._motion_task: asyncio.Task | None = None
        self._idle_prompt = "stand still"

    # ------------------------------------------------------------------
    # Tool
    # ------------------------------------------------------------------

    @function_tool()
    async def perform_motion(
        self,
        context: RunContext,
        motion_description: str,
        duration: str = "short",
    ):
        """Command the robot to perform a physical motion or gesture.

        Args:
            motion_description: A clear English description of the motion
                the robot should perform (e.g. 'wave right hand',
                'walk forward', 'clap both hands together').
            duration: How long the motion should last.
                'short' (5s) for quick gestures like wave, clap, bow.
                'long' (20s) for sustained motions like walking, dancing,
                boxing, or anything you want to keep going while you talk.
        """
        seconds = 20.0 if duration == "long" else 5.0
        logger.info(f"perform_motion → '{motion_description}' ({duration}, {seconds}s)")

        await self._engine.start_generation(motion_description)

        # Cancel previous motion→idle timer
        if self._motion_task and not self._motion_task.done():
            self._motion_task.cancel()
        self._motion_task = asyncio.get_event_loop().create_task(
            self._motion_then_idle(motion_description, seconds)
        )

        # Ensure 50 Hz streamer is running
        self._ensure_streamer()

        return f"Motion started: {motion_description}"

    # ------------------------------------------------------------------
    # Motion→idle lifecycle
    # ------------------------------------------------------------------

    async def _motion_then_idle(self, description: str, duration: float) -> None:
        """After *duration* seconds, switch generation to idle (stand still)."""
        try:
            await asyncio.sleep(duration)
            logger.info(f"Motion '{description}' done → idle")
            while True:
                await self._engine.start_generation(self._idle_prompt)
                await asyncio.sleep(4.0)
        except asyncio.CancelledError:
            pass

    # ------------------------------------------------------------------
    # 50 Hz data channel streamer
    # ------------------------------------------------------------------

    def _ensure_streamer(self) -> None:
        if self._streamer_task is None or self._streamer_task.done():
            self._streamer_task = asyncio.get_event_loop().create_task(
                self._stream_to_robot()
            )

    async def _stream_to_robot(self) -> None:
        """Continuous 50 Hz loop: pop frames → binary encode → publish."""
        interval = 1.0 / STREAM_HZ

        # Wait until enough frames are buffered
        logger.info(
            f"[Streamer] Waiting for {READY_THRESHOLD} buffered frames..."
        )
        while self._engine.queue_size() < READY_THRESHOLD:
            await asyncio.sleep(0.01)
        logger.info("[Streamer] Ready — streaming at 50 Hz")

        try:
            ctx = get_job_context()
        except RuntimeError:
            logger.warning("[Streamer] No job context (console mode?) — skipping")
            return

        while True:
            t0 = asyncio.get_event_loop().time()

            frame = self._engine.read_motion_frame()
            if frame is not None:
                payload = self._encode_frame(frame)
                try:
                    await ctx.room.local_participant.publish_data(
                        payload,
                        reliable=False,
                        topic="motion",
                        destination_identities=[ROBOT_IDENTITY],
                    )
                except Exception as exc:
                    logger.error(f"[Streamer] publish_data error: {exc}")

            # Maintain 50 Hz cadence
            elapsed = asyncio.get_event_loop().time() - t0
            sleep_time = max(0, interval - elapsed)
            await asyncio.sleep(sleep_time)

    @staticmethod
    def _encode_frame(frame: dict) -> bytes:
        """Encode a motion frame as 236-byte binary packet.

        Layout: 4-byte magic 'MF01' + 29 float32 dof_pos + 29 float32 dof_vel
        """
        dof_pos = frame["dof_pos"]
        dof_vel = frame["dof_vel"]
        # Flatten nested lists if present (decoder sometimes returns [[...]])
        if isinstance(dof_pos[0], list):
            dof_pos = dof_pos[0]
        if isinstance(dof_vel[0], list):
            dof_vel = dof_vel[0]
        return FRAME_MAGIC + struct.pack(f"<{29}f", *dof_pos) + struct.pack(
            f"<{29}f", *dof_vel
        )


# ---------------------------------------------------------------------------
# Server setup
# ---------------------------------------------------------------------------

server = AgentServer(
    num_idle_processes=1,               # only 1 process — avoid GPU contention
    initialize_process_timeout=120.0,   # model loading + warm-up takes ~15-30s
)


def prewarm(proc: JobProcess):
    """Load VAD + MotionEngine during prewarm (before any job starts)."""
    proc.userdata["vad"] = silero.VAD.load()

    # Load MotionEngine (Qwen + decoder + FSQ) — takes ~15-30s
    gpu_lock = asyncio.Lock()
    engine = MotionEngine(MODEL_PATH, DECODER_PATH, gpu_lock=gpu_lock)
    proc.userdata["motion_engine"] = engine
    proc.userdata["gpu_lock"] = gpu_lock


server.setup_fnc = prewarm


@server.rtc_session()
async def entrypoint(ctx: JobContext):
    engine: MotionEngine = ctx.proc.userdata["motion_engine"]
    gpu_lock: asyncio.Lock = ctx.proc.userdata["gpu_lock"]

    # Choose LLM: Qwen (default) or OpenAI fallback
    if USE_QWEN_CHAT:
        chat_llm = QwenLLM(engine.model, engine.tokenizer, gpu_lock)
        logger.info("Using Qwen 2.5 VL for chat")
    else:
        chat_llm = openai.LLM(model="gpt-4o")
        logger.info("Falling back to OpenAI GPT-4o for chat")

    session = AgentSession(
        stt=openai.STT(model="gpt-4o-transcribe"),
        llm=chat_llm,
        tts=openai.TTS(model="gpt-4o-mini-tts", voice="fable"),
        vad=ctx.proc.userdata["vad"],
    )

    agent = UnifiedRobotAgent(motion_engine=engine)
    await session.start(agent=agent, room=ctx.room)


if __name__ == "__main__":
    cli.run_app(server)
