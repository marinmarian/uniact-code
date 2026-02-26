"""
Dispatch the voice agent and open a browser page to talk to Darwin.

Usage:
    python livekit_connect.py          # uses ROOM_NAME from .env
    python livekit_connect.py myroom   # override room name
"""

import asyncio
import http.server
import socketserver
import sys
import os
import webbrowser

from dotenv import load_dotenv
from livekit import api

load_dotenv()

HTML_TEMPLATE = """\
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Darwin Voice</title>
<style>
  body {{ font-family: system-ui; background: #111; color: #eee;
         display: flex; flex-direction: column; align-items: center;
         justify-content: center; height: 100vh; margin: 0; }}
  #status {{ font-size: 1.4em; margin-bottom: 1em; }}
  button {{ font-size: 1.2em; padding: 0.6em 1.6em; border-radius: 8px;
            border: none; cursor: pointer; background: #2563eb; color: #fff; }}
  button:hover {{ background: #1d4ed8; }}
  #btn-disconnect {{ background: #dc2626; display: none; }}
  #btn-disconnect:hover {{ background: #b91c1c; }}
</style>
</head>
<body>
<div id="status">Click to connect</div>
<button id="btn-connect" onclick="start()">Talk to Darwin</button>
<button id="btn-disconnect" onclick="stop()">Disconnect</button>
<div id="audio-container"></div>
<script src="https://cdn.jsdelivr.net/npm/livekit-client@2.5.9/dist/livekit-client.umd.js"></script>
<script>
var LK_URL = "{url}";
var TOKEN = "{token}";
var room;

async function start() {{
  var status = document.getElementById("status");
  try {{
    status.textContent = "Requesting microphone...";
    await navigator.mediaDevices.getUserMedia({{ audio: true }});

    status.textContent = "Connecting to room...";
    room = new LivekitClient.Room();

    room.on(LivekitClient.RoomEvent.TrackSubscribed, function(track, publication, participant) {{
      console.log("Track subscribed:", track.kind);
      var el = track.attach();
      document.getElementById("audio-container").appendChild(el);
    }});

    room.on(LivekitClient.RoomEvent.Disconnected, function() {{
      status.textContent = "Disconnected";
      document.getElementById("btn-connect").style.display = "inline-block";
      document.getElementById("btn-disconnect").style.display = "none";
    }});

    await room.connect(LK_URL, TOKEN);
    console.log("Room connected");
    await room.localParticipant.setMicrophoneEnabled(true);
    console.log("Microphone enabled");

    status.textContent = "Connected — speak to Darwin";
    document.getElementById("btn-connect").style.display = "none";
    document.getElementById("btn-disconnect").style.display = "inline-block";
  }} catch (e) {{
    status.textContent = "Error: " + e.message;
    console.error(e);
  }}
}}

async function stop() {{
  if (room) await room.disconnect();
}}
</script>
</body>
</html>
"""


async def main():
    room_name = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("ROOM_NAME", "uniact-live")

    lk = api.LiveKitAPI(
        os.environ["LIVEKIT_URL"],
        os.environ["LIVEKIT_API_KEY"],
        os.environ["LIVEKIT_API_SECRET"],
    )

    url = os.environ["LIVEKIT_URL"]

    # Generate user token
    token = (
        api.AccessToken(os.environ["LIVEKIT_API_KEY"], os.environ["LIVEKIT_API_SECRET"])
        .with_identity("user-browser")
        .with_name("User")
        .with_grants(api.VideoGrants(
            room_join=True, room=room_name,
            can_publish=True, can_subscribe=True, can_publish_data=True,
        ))
        .to_jwt()
    )

    # Dispatch agent to the room
    dispatch = await lk.agent_dispatch.create_dispatch(
        api.CreateAgentDispatchRequest(room=room_name, agent_name="")
    )
    print(f"Room:    {room_name}")
    print(f"Agent:   dispatched (id: {dispatch.id})")

    await lk.aclose()

    # Serve HTML on localhost (required for mic permissions)
    html = HTML_TEMPLATE.format(url=url, token=token)
    html_bytes = html.encode()

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/favicon.ico":
                self.send_response(204)
                self.end_headers()
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(html_bytes)
        def log_message(self, *args):
            pass  # silence request logs

    class ThreadedServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
        daemon_threads = True

    port = 8080
    server = ThreadedServer(("127.0.0.1", port), Handler)
    print(f"Open:    http://localhost:{port}")
    webbrowser.open(f"http://localhost:{port}")
    print("Press Ctrl+C to stop")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped")


if __name__ == "__main__":
    asyncio.run(main())
