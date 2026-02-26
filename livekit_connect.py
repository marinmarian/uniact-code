"""
Dispatch the voice agent to a room.

Usage:
    python livekit_connect.py          # uses ROOM_NAME from .env
    python livekit_connect.py myroom   # override room name
"""

import asyncio
import sys
import os

from dotenv import load_dotenv
from livekit import api

load_dotenv()


async def main():
    room_name = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("ROOM_NAME", "uniact-live")

    lk = api.LiveKitAPI(
        os.environ["LIVEKIT_URL"],
        os.environ["LIVEKIT_API_KEY"],
        os.environ["LIVEKIT_API_SECRET"],
    )

    dispatch = await lk.agent_dispatch.create_dispatch(
        api.CreateAgentDispatchRequest(room=room_name, agent_name="")
    )
    print(f"Room:  {room_name}")
    print(f"Agent: dispatched (id: {dispatch.id})")

    await lk.aclose()


if __name__ == "__main__":
    asyncio.run(main())
