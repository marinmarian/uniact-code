# UniAct Setup & Run Instructions

## Architecture

Two machines working together:

- **AWS (server)** — runs the Qwen2.5-VL motion generation model on GPU
- **Mac (client)** — runs the MuJoCo physics simulation + RL tracking policy

---

## AWS Setup (one-time)

### Instance
- Type: NVIDIA L4 GPU, 23GB VRAM
- Storage: 150GB volume mounted at `/data`

### Steps
1. SSH into instance:
   ```bash
   ssh -i ~/.ssh/<your-key>.pem ubuntu@<your-aws-ip>
   ```
2. Clone repo and move to `/data`:
   ```bash
   git clone https://github.com/marinmarian/uniact-code.git
   cp -r ~/uniact-code /data/uniact/
   cd /data/uniact/uniact-code
   ```
3. Create venv and install dependencies:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   TMPDIR=/data/tmp pip install --cache-dir /data/pip-cache -r requirements_qwen.txt
   ```
4. Download checkpoints:
   ```bash
   pip install gdown
   gdown --folder "https://drive.google.com/drive/folders/1sh1IQdIjvnxx2s2did0vvZVP9DuCpoGE" -O /data/checkpoints
   ```
5. Configure `server.py`:
   - Set `MODEL_PATH = "/data/checkpoints/motion-generator"` in `main()`
   - Set `self.decoder = torch.jit.load('/data/checkpoints/decoder.pt')`

---

## Mac Setup (one-time)

1. Install dependencies:
   ```bash
   pip3 install -r requirements.txt
   pip3 install "livekit-agents[openai,silero]>=1.4"   # for voice agent
   ```
2. Download `policy.pt` from [Google Drive](https://drive.google.com/drive/folders/1sh1IQdIjvnxx2s2did0vvZVP9DuCpoGE) and place it in the project root. You can download manually from the browser, or use gdown:
   ```bash
   pip3 install gdown
   gdown --folder "https://drive.google.com/drive/folders/1sh1IQdIjvnxx2s2did0vvZVP9DuCpoGE" -O /tmp/checkpoints
   cp /tmp/checkpoints/policy.pt .
   ```
3. `configs/g1_ref_real.yaml` already points to `./policy.pt` — no changes needed as long as you run from the project root.
4. Copy `.env.example` to `.env` and fill in your credentials:
   ```bash
   cp .env.example .env
   # Edit .env with your LiveKit and OpenAI API keys
   ```

---

## Running (every time)

### Option A: File / CLI mode

#### Terminal 1 — AWS server
```bash
ssh -i ~/.ssh/<your-key>.pem ubuntu@<your-aws-ip>
cd /data/uniact/uniact-code
source venv/bin/activate
python server.py
```

#### Terminal 2 — SSH tunnel (Mac)
```bash
ssh -L 8000:localhost:8000 -i ~/.ssh/<your-key>.pem ubuntu@<your-aws-ip>
```
Keep this open — it forwards port 8000 from Mac to AWS.

#### Terminal 3 — Simulation (Mac)
```bash
cd /path/to/uniact-code
mjpython robot_client.py                    # File mode (reads text.jsonl)
mjpython robot_client.py --use_commandline  # Interactive CLI mode
```

### Option B: Voice agent mode (speak to control the robot)

#### Terminals 1 & 2 — same as above (AWS server + SSH tunnel)

#### Terminal 3 — Voice agent (Mac)
```bash
python livekit_voice_agent.py dev
```

#### Terminal 4 — Robot client + bridge (Mac)
```bash
mjpython robot_client.py --use_livekit
```
The robot bridge auto-discovers the playground room — no manual room config needed.

#### Browser — Connect via playground
Go to https://agents-playground.livekit.io and connect. The robot will auto-join the same room.

---

## Controlling the Robot

### Voice mode
Speak naturally in the browser playground: "wave hello", "walk forward", "do a jumping jack".

### File mode
Edit `text.jsonl` to schedule motion commands before running:
```json
{"frame": 0, "text": "walk forward"}
{"frame": 200, "text": "turn left"}
{"frame": 400, "text": "dance"}
```

- `frame` — when to send the command (at 50Hz, 200 frames = 4 seconds)
- `text` — natural language motion description

### CLI mode
```bash
mjpython robot_client.py --use_commandline
```
Then type commands like `start walk forward`, `stop`, `quit`.

---

## Notes
- Always start the AWS server before running the Mac client
- Keep the SSH tunnel open while simulating
- The MuJoCo viewer window will open automatically on Mac
- `RECORD_VIDEO = False` in `robot_client.py` — uses live viewer on Mac
- For voice mode, you need LiveKit + OpenAI credentials in `.env`
