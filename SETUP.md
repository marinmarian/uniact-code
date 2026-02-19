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
1. SSH into instance
2. Clone repo and move to `/data`:
   ```bash
   git clone https://github.com/jnnan/uniact-code.git
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
   - Set `MODEL_PATH = "/data/checkpoints/motion-genorator"` in `main()`
   - Set `self.decoder = torch.jit.load('/data/checkpoints/decoder.pt')`

---

## Mac Setup (one-time)

1. Install dependencies:
   ```bash
   pip3 install mujoco imageio torch scipy omegaconf joblib numpy matplotlib easydict
   ```
2. Copy `policy.pt` from AWS:
   ```bash
   scp -i ~/.ssh/darwin_keys.pem ubuntu@18.171.160.139:/data/checkpoints/policy.pt /Users/marinmarian/Work/humanoid_control/uniact-code/
   ```
3. Set policy path in `configs/g1_ref_real.yaml`:
   ```yaml
   policy_path: "/Users/marinmarian/Work/humanoid_control/uniact-code/policy.pt"
   ```

---

## Running (every time)

### Terminal 1 — AWS server
```bash
ssh -i ~/.ssh/darwin_keys.pem ubuntu@18.171.160.139
cd /data/uniact/uniact-code
source venv/bin/activate
python server.py
```

### Terminal 2 — SSH tunnel (Mac)
```bash
ssh -L 8000:localhost:8000 -i ~/.ssh/darwin_keys.pem ubuntu@18.171.160.139
```
Keep this open — it forwards port 8000 from Mac to AWS.

### Terminal 3 — Simulation (Mac)
```bash
cd /Users/marinmarian/Work/humanoid_control/uniact-code
mjpython robot_client.py
```

---

## Controlling the Robot

Edit `text.jsonl` to schedule motion commands before running:
```json
{"frame": 0, "text": "walk forward"}
{"frame": 200, "text": "turn left"}
{"frame": 400, "text": "dance"}
```

- `frame` — when to send the command (at 50Hz, 200 frames = 4 seconds)
- `text` — natural language motion description

Or use interactive mode:
```bash
mjpython robot_client.py --use_commandline
```
Then type commands like `start walk forward`, `stop`, `quit`.

---

## Notes
- Always start the AWS server before running the Mac client
- Keep both Terminal 1 and Terminal 2 open while simulating
- The MuJoCo viewer window will open automatically on Mac
- `RECORD_VIDEO = False` in `robot_client.py` — uses live viewer on Mac
