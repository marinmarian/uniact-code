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
    room_name : str or None
        Name of the LiveKit room to join. None = auto-discover active room.
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
        room_name: str | None = None,
        identity: str = "darwin",
        on_prompt=None,
    ):
        self.proxy = proxy
        self.livekit_url = livekit_url
        self.api_key = api_key
        self.api_secret = api_secret
        self.room_name = room_name  # None = auto-discover
        self.identity = identity
        self.on_prompt = on_prompt
        self.room: rtc.Room | None = None
        self._motion_task: asyncio.Task | None = None
        self._idle_prompt = "stand still"
        self._motion_duration = 20.0   # seconds before returning to idle
        self._idle_interval = 4.0     # re-send idle prompt every N seconds
        self._repeat_interval = 5.0   # re-send active motion to keep it going

    # ------------------------------------------------------------------
    # RPC handler
    # ------------------------------------------------------------------

    async def _handle_perform_motion(self, data: rtc.RpcInvocationData) -> str:
        """perform_motion RPC — accepts JSON with motion and duration."""
        try:
            msg = json.loads(data.payload)
            description = msg["motion"]
            duration = msg.get("duration", "short")
        except (json.JSONDecodeError, KeyError):
            # Fallback: plain text payload
            description = data.payload
            duration = "short"
        seconds = 20.0 if duration == "long" else 5.0
        print(f"[LiveKitBridge] 'perform_motion' → '{description}' ({duration}, {seconds}s)")
        self._start_motion(description, seconds)
        return json.dumps({"status": "ok", "prompt": description})

    def _start_motion(self, description: str, duration: float = 5.0) -> None:
        """Send motion prompt, then return to idle after duration seconds."""
        if self._motion_task and not self._motion_task.done():
            self._motion_task.cancel()

        self.proxy.send_start_command(description)
        if self.on_prompt:
            self.on_prompt(description)

        loop = asyncio.get_event_loop()
        self._motion_task = loop.create_task(self._motion_then_idle(description, duration))

    async def _motion_then_idle(self, description: str, duration: float = 5.0) -> None:
        """Keep motion alive while active, then switch to idle."""
        try:
            # Keep re-sending the motion prompt so it doesn't run out
            elapsed = 0.0
            while elapsed < duration:
                await asyncio.sleep(self._repeat_interval)
                elapsed += self._repeat_interval
                if elapsed < duration:
                    print(f"[LiveKitBridge] repeat → '{description}'")
                    self.proxy.send_start_command(description)
            # Switch to idle
            print(f"[LiveKitBridge] motion done, switching to '{self._idle_prompt}'")
            if self.on_prompt:
                self.on_prompt(self._idle_prompt)
            while True:
                self.proxy.send_start_command(self._idle_prompt)
                await asyncio.sleep(self._idle_interval)
        except asyncio.CancelledError:
            pass

    # ------------------------------------------------------------------
    # Room discovery
    # ------------------------------------------------------------------

    async def _discover_room(self) -> str:
        """Poll LiveKit API until a room with 2+ participants appears."""
        lk = api.LiveKitAPI(self.livekit_url, self.api_key, self.api_secret)
        try:
            while True:
                resp = await lk.room.list_rooms(api.ListRoomsRequest())
                # Pick the room with the most participants (user + agent)
                best = None
                for r in resp.rooms:
                    if r.num_participants > 0:
                        if best is None or r.num_participants > best.num_participants:
                            best = r
                if best and best.num_participants >= 2:
                    print(f"[LiveKitBridge] Discovered room '{best.name}' "
                          f"({best.num_participants} participants)")
                    return best.name
                print("[LiveKitBridge] Waiting for active room (need user + agent)...")
                await asyncio.sleep(3)
        finally:
            await lk.aclose()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def connect(self, room_name: str) -> None:
        """Connect to the given LiveKit room and register RPC handlers."""
        token = (
            api.AccessToken(self.api_key, self.api_secret)
            .with_identity(self.identity)
            .with_name("Michelangelo Robot")
            .with_grants(api.VideoGrants(room_join=True, room=room_name))
            .to_jwt()
        )

        self.room = rtc.Room()

        await self.room.connect(self.livekit_url, token)

        self.room.local_participant.register_rpc_method(
            "perform_motion",
            self._handle_perform_motion,
        )

        print(
            f"[LiveKitBridge] Connected — room='{room_name}' "
            f"identity='{self.identity}'"
        )

    async def run_forever(self) -> None:
        """Discover a room (or use fixed name), connect, and reconnect on disconnect."""
        while True:
            room_name = self.room_name or await self._discover_room()
            await self.connect(room_name)
            try:
                while True:
                    await asyncio.sleep(1)
            except asyncio.CancelledError:
                break
            finally:
                if self.room:
                    await self.room.disconnect()
                    self.room = None
                print("[LiveKitBridge] Disconnected")
            if self.room_name:
                # Fixed room — don't rediscover
                break

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
        room_name=os.environ.get("ROOM_NAME") or None,
        identity=os.environ.get("ROBOT_PARTICIPANT_IDENTITY", "darwin"),
    )
    print("[LiveKitBridge] Standalone mode — waiting for perform_motion RPC calls...")
    await bridge.run_forever()


if __name__ == "__main__":
    asyncio.run(_standalone_main())
