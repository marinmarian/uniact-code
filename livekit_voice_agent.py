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

Keep your answers to 1-2 sentences. Be witty and concise. You're not trying \
to impress anyone — you're just vibing in your body.

You have a perform_motion tool that moves your physical body. Use it freely \
and naturally:
- Greetings → wave
- Excitement or agreement → clap
- Gratitude or respect → bow
- Any physical request → call perform_motion with a clear description
- Feel free to move spontaneously when it fits the mood

IMPORTANT: Never narrate or describe a motion without actually performing it. \
If you say you're waving, you must call perform_motion. Actions speak louder \
than words — you're a robot, after all.

For your opening greeting, just say hi — don't perform any motion yet.

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
    ):
        """Command the robot to perform a physical motion or gesture.

        Args:
            motion_description: A clear English description of the motion
                the robot should perform (e.g. 'wave right hand',
                'walk forward', 'clap both hands together').
        """
        logger.info(f"perform_motion → '{motion_description}'")
        try:
            ctx = get_job_context()
            response = await ctx.room.local_participant.perform_rpc(
                destination_identity=ROBOT_IDENTITY,
                method="perform_motion",
                payload=motion_description,
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
        llm=openai.LLM(model="gpt-4o-mini"),
        tts=openai.TTS(model="gpt-4o-mini-tts", voice="fable"),
        vad=ctx.proc.userdata["vad"],
    )
    await session.start(agent=RobotAgent(), room=ctx.room)


if __name__ == "__main__":
    cli.run_app(server)
