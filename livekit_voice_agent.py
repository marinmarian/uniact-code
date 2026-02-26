"""
LiveKit Voice Agent for Robot Control

Voice pipeline agent that listens to natural language commands and forwards
them to the robot participant via RPC.  The robot side is handled by
livekit_bridge.py (started automatically by robot_client.py --use_livekit).

Usage:
    python livekit_voice_agent.py dev       # development mode (hot reload)
    python livekit_voice_agent.py console   # test locally in terminal
    python livekit_voice_agent.py start     # production
"""

import json
import logging

logging.basicConfig(level=logging.INFO)

import os

from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    JobProcess,
    RunContext,
    ToolError,
    cli,
    function_tool,
    get_job_context,
)
from livekit.plugins import openai, silero

load_dotenv()

logger = logging.getLogger("voice-agent")

ROBOT_IDENTITY = os.environ.get("ROBOT_PARTICIPANT_IDENTITY", "darwin")

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


class RobotAgent(Agent):
    def __init__(self) -> None:
        super().__init__(instructions=SYSTEM_PROMPT)

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
        logger.info(f"perform_motion → '{motion_description}' ({duration})")
        payload = json.dumps({"motion": motion_description, "duration": duration})
        try:
            ctx = get_job_context()
            response = await ctx.room.local_participant.perform_rpc(
                destination_identity=ROBOT_IDENTITY,
                method="perform_motion",
                payload=payload,
                response_timeout=3.0,
            )
            return f"Robot acknowledged: {response}"
        except RuntimeError:
            # Console mode — no room available
            logger.info(f"perform_motion (no room) → '{motion_description}'")
            return f"Motion queued: {motion_description}"
        except Exception as exc:
            logger.error(f"RPC failed: {exc}")
            return f"Motion sent (robot offline): {motion_description}"


# ---------------------------------------------------------------------------
# Server setup
# ---------------------------------------------------------------------------

server = AgentServer()


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


server.setup_fnc = prewarm


@server.rtc_session()
async def entrypoint(ctx: JobContext):
    session = AgentSession(
        stt=openai.STT(model="gpt-4o-transcribe"),
        llm=openai.LLM(model="gpt-4o"),
        tts=openai.TTS(model="gpt-4o-mini-tts", voice="fable"),
        vad=ctx.proc.userdata["vad"],
    )
    await session.start(agent=RobotAgent(), room=ctx.room)


if __name__ == "__main__":
    cli.run_app(server)
