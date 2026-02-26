# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

UniAct is a real-time text-to-motion system for the Unitree G1 humanoid robot (29 DOF). It converts text prompts into physically-simulated whole-body motion via a 3-tier pipeline: **Server** (GPU, Qwen 2.5 VL LLM) → **Proxy** (token buffer/interpolator) → **Robot Client** (50Hz MuJoCo control loop).

Paper: arXiv 2512.24321

## Commands

### Running the system

```bash
# Server (GPU machine) - generates motion tokens via Qwen LLM
python server.py

# Client (local Mac) - MuJoCo simulation
mjpython robot_client.py                    # File mode (reads text.jsonl)
mjpython robot_client.py --use_commandline  # Interactive CLI mode
mjpython robot_client.py --use_livekit      # LiveKit voice agent mode

# Headless (AWS)
MUJOCO_GL=egl python robot_client.py
```

### LiveKit voice agent setup
```bash
# Terminal 1: SSH into AWS, run server.py
# Terminal 2: SSH tunnel: ssh -L 8000:localhost:8000 -i <key> ubuntu@<aws-ip>
# Terminal 3: Voice agent (Mac or anywhere with OPENAI_API_KEY)
python livekit_voice_agent.py dev
# Terminal 4: Robot client + bridge (Mac)
mjpython robot_client.py --use_livekit
# Terminal 5: Generate playground token + dispatch agent
python livekit_connect.py
# Browser: paste URL + token into https://agents-playground.livekit.io (Custom connect)
```

### Distributed setup (server on AWS, client on Mac)
```bash
# Terminal 1: SSH into AWS, run server.py
# Terminal 2: SSH tunnel: ssh -L 8000:localhost:8000 -i <key> ubuntu@<aws-ip>
# Terminal 3: Run robot_client.py locally
```

### Benchmarking
```bash
python3 benchmark_t2m.py  # Evaluates 66 text-to-motion samples in t2m/
```

### Dependencies
```bash
pip install -r requirements.txt                      # Client (mujoco, torch, livekit)
pip install -r requirements_qwen.txt                 # Server (Qwen VL, CUDA, transformers)
pip install "livekit-agents[openai,silero]>=1.4"     # Voice agent (STT/LLM/TTS)
```

Python 3.11+ required.

## Architecture

```
text.jsonl / CLI / Voice Agent
    ↓
server.py (GPU)     ← Qwen 2.5 VL → FSQ motion tokens (autoregressively)
    ↓ TCP :8000
proxy.py (buffer)   ← Token queue, sliding window, 40-frame interpolation on prompt switch
    ↓ local
robot_client.py     ← 50Hz: 151D obs → RL policy (policy.pt) → 29D action → PD control → MuJoCo 1kHz

Voice agent flow (LiveKit mode):
User (browser mic) → LiveKit Cloud → livekit_voice_agent.py (STT→LLM→TTS)
    → perform_motion RPC → livekit_bridge.py → proxy.send_start_command() → server.py → MuJoCo
```

### Key data flow
1. Text prompt → Qwen LLM generates FSQ token IDs (discrete codes from 15,360-entry codebook)
2. Proxy decodes tokens → joint pos/vel via TorchScript decoder
3. Client builds 151D observation (ref pose + current state + IMU + gravity + prev action)
4. RL policy outputs 29D joint targets; PD control computes torques; MuJoCo steps 20x at 1kHz per 50Hz tick

### Critical: Joint ordering
Two orderings exist: **BYD** (training) and **MuJoCo** (simulation). Index mapping tables (`byd_joint_to_mujoco_joint`, `mujoco_joint_to_byd_joint`) exist in both `server.py` and `robot_client.py`. These must stay synchronized.

## Key Files

| File | Role |
|------|------|
| `robot_client.py` | Main controller: G1 class (PD control, MuJoCo), DeployNode (50Hz loop) |
| `server.py` | MotionServer: Qwen LLM inference, FSQ decode, TCP socket |
| `proxy.py` | MotionProxy: token buffer, interpolation, queue management |
| `livekit_voice_agent.py` | Voice pipeline agent: STT/LLM/TTS via OpenAI, perform_motion RPC tool |
| `livekit_bridge.py` | LiveKitBridge: perform_motion RPC → proxy (daemon thread, asyncio) |
| `livekit_connect.py` | Helper: generates playground token + dispatches agent to room |
| `infer_robot.py` | LLM inference utilities: KV-cache, special token handling (IDs start at 129,627) |
| `infer_fsq_ar.py` | TokenDecoder: FSQ autoregressive decode with 8-code sliding window |
| `fsq.py` | FSQ module: levels [8,8,8,6,5] = 15,360 codes, straight-through gradients |
| `configs/g1_ref_real.yaml` | Main config: model paths, sim params (dt=0.001, decimation=20) |

## Configuration

### Paths to configure before running
- `server.py`: `MODEL_PATH` (Qwen weights dir) and decoder path (`torch.jit.load`)
- `configs/g1_ref_real.yaml`: `xml_path`, `policy_path`, `motion_file`
- `.env` (from `.env.example`): LiveKit credentials + `OPENAI_API_KEY` (gitignored)

### Simulation flags (top of `robot_client.py`)
- `RECORD_VIDEO`: True for headless AWS, False for Mac (live viewer). Outputs `simulation_output.mp4`
- `ONLINE_MOTION`: True = server-generated, False = pre-recorded pkl
- `DEBUG`/`SIM`: True = MuJoCo sim, False = real hardware

### Video recording (`RECORD_VIDEO = True`)
- 1280x720 H.264 (yuv420p) at 50fps, QuickTime-compatible
- Tracking camera follows the robot pelvis (azimuth 135, elevation -20, distance 3.0)
- Current text prompt overlaid at bottom center (white text, black outline) via cv2
- Requires `opencv-python` and `imageio[ffmpeg]`

### Text schedule files
- `text.jsonl`: Main prompt schedule read by the client (tracked in git)
- `text_*.jsonl`: Preset schedules (actions, gestures, running, walking) — gitignored
- Format: `{"frame": <int>, "text": "<prompt>"}` per line, sorted by frame

### Runtime config (robot_client.py `main()`)
- `ready_threshold` (30): tokens buffered before starting
- `buffer_threshold` (50): request more when queue drops below
- `keep_tokens_on_new_instruction` (30): retained on prompt change
- `keep_tokens_for_generate` (48): context tokens for server (must match server decoder overlap)

## Conventions

- Heavy use of `@torch.jit.script` for performance-critical rotation/quaternion ops
- Quaternion format: w-last in `torch_jit_utils.py`, varies in `isaac_utils/`
- Threading: proxy uses `threading.Lock()` for queue sync; LiveKit runs in daemon thread with own asyncio loop
- No formal test suite; `benchmark_t2m.py` serves as the evaluation script
- Observation normalization uses training statistics (obs_mean, obs_std) - critical for policy inference
- 50Hz hard deadline: physics stepped 20x per control tick at 1kHz

## Model Checkpoints

Downloaded from [Google Drive](https://drive.google.com/drive/folders/1sh1IQdIjvnxx2s2did0vvZVP9DuCpoGE):
- `motion-generator/` — fine-tuned Qwen 2.5 VL weights
- `decoder.pt` — FSQ decoder (TorchScript)
- `policy.pt` — RL tracking policy (TorchScript)
