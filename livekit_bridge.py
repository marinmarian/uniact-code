"""
LiveKit <-> UniAct Bridge

Joins the LiveKit room as the robot participant (robot123987), receives
perform_motion RPC calls from the voice agent, and forwards the motion
description to the UniAct MotionProxy.

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


class LiveKitBridge:
    """
    LiveKit participant that receives perform_motion RPC calls and forwards
    them to the UniAct MotionProxy.

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
        voice agent config (default: "darwin").
    on_prompt : callable, optional
        Callback invoked with the text description whenever a new motion
        prompt is received (used to update the video overlay).
    """

    def __init__(
        self,
        proxy,
        livekit_url: str,
        api_key: str,
        api_secret: str,
        room_name: str,
        identity: str = "darwin",
        on_prompt=None,
    ):
        self.proxy = proxy
        self.livekit_url = livekit_url
        self.api_key = api_key
        self.api_secret = api_secret
        self.room_name = room_name
        self.identity = identity
        self.on_prompt = on_prompt
        self.room: rtc.Room | None = None

    # ------------------------------------------------------------------
    # RPC handler
    # ------------------------------------------------------------------

    async def _handle_perform_motion(self, data: rtc.RpcInvocationData) -> str:
        """perform_motion RPC — accepts any text description."""
        description = data.payload
        print(f"[LiveKitBridge] 'perform_motion' → '{description}'")
        self.proxy.send_start_command(description)
        if self.on_prompt:
            self.on_prompt(description)
        return json.dumps({"status": "ok", "prompt": description})

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

        self.room.local_participant.register_rpc_method(
            "perform_motion",
            self._handle_perform_motion,
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
        identity=os.environ.get("ROBOT_PARTICIPANT_IDENTITY", "darwin"),
    )
    print("[LiveKitBridge] Standalone mode — waiting for perform_motion RPC calls...")
    await bridge.run_forever()


if __name__ == "__main__":
    asyncio.run(_standalone_main())
