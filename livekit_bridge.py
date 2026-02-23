"""
LiveKit <-> UniAct Bridge

Joins the LiveKit room as the robot participant (robot123987), receives RPC
gesture commands from the Darwin voice agent, translates them to plain-English
motion prompts, and forwards them to the UniAct MotionProxy.

Standalone test mode (no real robot):
    python livekit_bridge.py

Normal usage: started automatically by robot_client.py --use_livekit
"""

import asyncio
import json
import os
import threading

from dotenv import load_dotenv
from livekit import api, rtc

load_dotenv()

# ---------------------------------------------------------------------------
# Gesture name → UniAct text prompt mapping
# ---------------------------------------------------------------------------

GESTURE_PROMPTS: dict[str, str] = {
    "shake_hand":    "extend right arm forward for a handshake",
    "face_wave":     "wave right hand at face level",
    "high_wave":     "raise both arms and wave enthusiastically",
    "clap":          "clap both hands together",
    "high_five":     "raise right hand high for a high five",
    "hug":           "open both arms wide as if hugging someone",
    "right_kiss":    "lean head gently to the right",
    "left_kiss":     "lean head gently to the left",
    "two_hand_kiss": "raise both hands and blow a kiss",
    "hands_up":      "raise both hands above the head",
    "right_hand_up": "raise right hand straight up",
    "right_heart":   "make a heart shape gesture with both hands",
    "x_ray":         "stand straight and raise right hand to volunteer",
    "reject":        "wave hand side to side in refusal",
    "release_hand":  "lower arms slowly and return to standing position",
}


# ---------------------------------------------------------------------------
# Bridge
# ---------------------------------------------------------------------------

class LiveKitBridge:
    """
    LiveKit participant that bridges Darwin RPC calls → UniAct motion prompts.

    Parameters
    ----------
    proxy : MotionProxy
        A connected MotionProxy instance (owned by robot_client.py).
    livekit_url, api_key, api_secret : str
        LiveKit server credentials.
    room_name : str
        Name of the LiveKit room to join.
    identity : str
        Participant identity — must match ROBOT_PARTICIPANT_IDENTITY in the
        voice agent config (default: "robot123987").
    """

    def __init__(
        self,
        proxy,
        livekit_url: str,
        api_key: str,
        api_secret: str,
        room_name: str,
        identity: str = "robot123987",
    ):
        self.proxy = proxy
        self.livekit_url = livekit_url
        self.api_key = api_key
        self.api_secret = api_secret
        self.room_name = room_name
        self.identity = identity
        self.room: rtc.Room | None = None

    # ------------------------------------------------------------------
    # RPC handlers
    # ------------------------------------------------------------------

    def _make_gesture_handler(self, gesture_name: str):
        """Return an async RPC handler closure for the given gesture."""

        async def handler(data: rtc.RpcInvocationData) -> str:
            prompt = GESTURE_PROMPTS[gesture_name]
            print(f"[LiveKitBridge] '{gesture_name}' → '{prompt}'")
            self.proxy.send_start_command(prompt)
            return json.dumps({
                "status": "ok",
                "gesture": gesture_name,
                "prompt": prompt,
            })

        return handler

    async def _handle_identify_person(self, data: rtc.RpcInvocationData) -> str:
        """identify_person is a vision call — no motion generated."""
        print("[LiveKitBridge] 'identify_person' received — returning empty list")
        return json.dumps({"identities": []})

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        """Connect to the LiveKit room and register all RPC method handlers."""
        token = (
            api.AccessToken(self.api_key, self.api_secret)
            .with_identity(self.identity)
            .with_name("Darwin Robot")
            .with_grants(api.VideoGrants(room_join=True, room=self.room_name))
            .to_jwt()
        )

        self.room = rtc.Room()

        await self.room.connect(self.livekit_url, token)

        for gesture_name in GESTURE_PROMPTS:
            self.room.local_participant.register_rpc_method(
                gesture_name,
                self._make_gesture_handler(gesture_name),
            )

        self.room.local_participant.register_rpc_method(
            "identify_person",
            self._handle_identify_person,
        )

        print(
            f"[LiveKitBridge] Connected — room='{self.room_name}' "
            f"identity='{self.identity}'"
        )

    async def run_forever(self) -> None:
        """Connect and keep running until cancelled."""
        await self.connect()
        try:
            while True:
                await asyncio.sleep(1)
        except asyncio.CancelledError:
            pass
        finally:
            if self.room:
                await self.room.disconnect()
            print("[LiveKitBridge] Disconnected")

    def start_in_background(self) -> threading.Thread:
        """
        Start the bridge in a daemon thread with its own event loop.
        Called by robot_client.py after the proxy is connected.
        """

        def _run():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(self.run_forever())
            except Exception as exc:
                print(f"[LiveKitBridge] Fatal error: {exc}")
            finally:
                loop.close()

        thread = threading.Thread(target=_run, daemon=True, name="livekit-bridge")
        thread.start()
        return thread


# ---------------------------------------------------------------------------
# Standalone entry point — test without a real robot
# ---------------------------------------------------------------------------

async def _standalone_main() -> None:
    class _FakeProxy:
        def send_start_command(self, prompt: str) -> None:
            print(f"[FakeProxy] Would send to UniAct: '{prompt}'")

    bridge = LiveKitBridge(
        proxy=_FakeProxy(),
        livekit_url=os.environ["LIVEKIT_URL"],
        api_key=os.environ["LIVEKIT_API_KEY"],
        api_secret=os.environ["LIVEKIT_API_SECRET"],
        room_name=os.environ.get("ROOM_NAME", "darwin-robot"),
        identity=os.environ.get("ROBOT_PARTICIPANT_IDENTITY", "robot123987"),
    )
    print("[LiveKitBridge] Standalone mode — waiting for RPC calls from Darwin...")
    await bridge.run_forever()


if __name__ == "__main__":
    asyncio.run(_standalone_main())
