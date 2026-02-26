# UniAct — Unified Motion Generation for Humanoid Robots

This repo implements a real-time text-to-motion system for the **Unitree G1 humanoid robot**. You give it a text prompt like "perform a golf swing" and it generates full-body motion executed in MuJoCo simulation or on real hardware.

Paper: [arXiv 2512.24321](https://www.arxiv.org/pdf/2512.24321)

---

## Table of Contents

1. [How It Works](#how-it-works)
2. [Architecture](#architecture-3-tier-pipeline)
3. [Control Loop](#control-loop-simplified)
4. [RL Policy Network](#rl-policy-network)
5. [Running on AWS](#running-on-aws)
6. [Tunable Parameters](#tunable-parameters)
7. [Commands Reference](#commands-reference)
8. [Key Components](#key-components)
9. [Data Files](#data-files)
10. [Benchmarking](#benchmarking)

---

## How It Works

The system converts **text → motion tokens → joint angles → physics simulation** in real time.

1. You type a prompt like `start walk forward`
2. A fine-tuned **Qwen 2.5 VL** language model generates motion tokens autoregressively (one token at a time, streamed)
3. Each token is an **FSQ code** — a discrete index into a 15,360-entry codebook that maps to a 5D vector
4. A **decoder** converts batches of FSQ codes into 29 joint positions + velocities for the G1 robot
5. An **RL policy** takes the decoded reference motion + the robot's current state and outputs corrected joint targets that account for physics/balance
6. **PD control** converts joint targets into torques, which MuJoCo simulates at 1kHz

The key insight: the LLM generates "ideal" motion, but real physics requires corrections. The RL policy bridges that gap.

---

## Architecture: 3-Tier Pipeline

```
  User (browser mic)
       │ audio
       ▼
  LiveKit Cloud
       │
       ▼
  livekit_voice_agent.py        text.jsonl /
  (STT → LLM → TTS)            --use_commandline
       │ perform_motion RPC           │
       ▼                              │
  livekit_bridge.py ──────────────────┘
       │ text prompt
       ▼
┌─────────┐     TCP      ┌─────────┐     Local     ┌──────────────┐
│  Server  │ ──────────── │  Proxy  │ ──────────── │ Robot Client  │
│  (GPU)   │   tokens     │ (buffer)│   motion      │ (MuJoCo/HW)  │
└─────────┘              └─────────┘               └──────────────┘
```

### `server.py` — Motion Token Generator
- Runs on a GPU machine
- Loads **Qwen 2.5 VL** (a vision-language model fine-tuned for motion generation)
- Takes text prompts → generates FSQ motion token sequences autoregressively
- Decodes tokens into joint pos/vel using FSQ quantizer + TorchScript decoder
- Uses `infer_robot.py` for generation (KV-cache, position encoding, token parsing)

### `livekit_voice_agent.py` — Voice Pipeline Agent
- LiveKit Agents SDK (1.x) voice pipeline: STT (OpenAI Whisper) → LLM (GPT-4o-mini) → TTS (OpenAI)
- Silero VAD for voice activity detection
- Single function tool: `perform_motion(motion_description)` — sends RPC to the robot participant
- Run with `python livekit_voice_agent.py dev`
- Requires `OPENAI_API_KEY` in `.env`

### `livekit_bridge.py` — Voice Agent Bridge
- Joins a LiveKit room as the robot participant (`robot123987` by default)
- Registers `perform_motion` RPC handler — accepts free-form text descriptions
- Calls `proxy.send_start_command(description)` to trigger motion
- Calls `on_prompt` callback to update video overlay text
- Runs in a daemon thread with its own asyncio event loop — does not block the main control loop
- Credentials loaded from `.env` (gitignored)

### `livekit_connect.py` — Connection Helper
- Generates a playground token for the LiveKit room
- Dispatches the voice agent to the room via LiveKit API
- Run with `python livekit_connect.py`

### `proxy.py` — Token Buffer/Interpolator
- Sits between server and client
- Maintains a **sliding window** of tokens
- Handles buffering, interpolation between discrete tokens, and queue management
- When switching prompts, interpolates 40 frames between old and new motion for smooth transitions
- Logs latency to `proxy_time_hot.txt`

### `robot_client.py` — Controller (the main file)
- Runs the **50Hz real-time control loop**
- Two phases:
  1. **Stand-up** — smooth interpolation from current pose to initial stance (500 steps)
  2. **Policy running** — RL policy inference + PD control at 50Hz

---

## Control Loop (Simplified)

The control loop in `robot_client.py` runs at **50Hz** (20ms per cycle). Each tick:

1. **Read state** — grab the robot's current joint angles, joint velocities, and IMU (orientation + angular velocity) from MuJoCo
2. **Get reference motion** — ask the proxy "where should the robot be right now?" — this returns target joint positions/velocities decoded from the LLM's motion tokens
3. **Build observation** — pack 151 numbers into a vector: the reference pose, the current pose, gravity direction, angular velocity, and what the policy did last step
4. **Run the policy** — feed the 151D observation into the RL neural network, get back 29 numbers (one per joint) — these are the desired joint angles
5. **Apply PD control** — for each joint, compute a torque:
   ```
   torque = P_gain * (desired_angle - current_angle)
          + D_gain * (0 - current_velocity)
   ```
   This is like a spring (P) pulling toward the target plus a damper (D) to prevent overshooting.
6. **Step physics** — apply those torques and run MuJoCo physics 20 times at 1kHz (20 x 1ms = 20ms = one control step)
7. **Render** — capture a frame for video

In short: **"where should I be?" → "where am I?" → "how do I get there?" → push → repeat.**

---

## RL Policy Network

The policy is a trained neural net that learned through reinforcement learning how to make the robot track reference motions without falling over.

**What it is:**
- A TorchScript model loaded from `policy.pt`
- Input: 151 floats (the observation vector)
- Output: 29 floats (one action per joint)

**What it learned during training:**
- Rewarded for: matching reference poses, staying balanced, smooth movements
- Penalized for: falling over, large torques, jerky motions

**Why it's needed:**
The reference motion from the LLM is just "ideal" joint angles — it doesn't account for physics, gravity, inertia, or balance. If you naively set joints to those angles, the robot falls over.

The RL policy acts as a **translator** between "where I should be" and "what I should actually do" given real physics. It makes small corrections — leaning slightly, adjusting timing, compensating for momentum — so the robot stays upright while following the motion.

**Analogy:** The LLM is like a choreographer saying "move your arm here." The RL policy is the dancer's muscles and reflexes that actually execute it without tripping.

---

## Running on AWS

### Prerequisites

- AWS instance with GPU (for server — needs to run Qwen 2.5 VL)
- Python 3.11+
- CUDA + cuDNN installed

### Model Checkpoints

Download from [Google Drive](https://drive.google.com/drive/folders/1sh1IQdIjvnxx2s2did0vvZVP9DuCpoGE?usp=sharing):
- `motion-generator/` — Qwen 2.5 VL fine-tuned weights
- `decoder.pt` — FSQ decoder (TorchScript)
- `policy.pt` — RL tracking policy (TorchScript)

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Configure Paths

Edit `configs/g1_ref_real.yaml`:
```yaml
xml_path: "unitree_description/mjcf/g1.xml"
policy_path: "/path/to/policy.pt"
motion_file: "data/motion_data/raw_walking_g1_deploy.pkl"
```

Edit `server.py`:
```python
# Line ~147: decoder path
self.decoder = torch.jit.load('/path/to/decoder.pt')

# Line ~622: model path
MODEL_PATH = "/path/to/motion-generator"
```

### Step 3: Start the Server

```bash
python server.py
```

Wait for "Model loaded successfully!" and "Server listening on ...".

### Step 4: Start the Client (new terminal)

For headless AWS (no display):
```bash
MUJOCO_GL=egl python robot_client.py --use_text_file
```

Or for interactive mode:
```bash
MUJOCO_GL=egl python robot_client.py --use_commandline
```

`MUJOCO_GL=egl` is **required** on headless servers — it tells MuJoCo to render using EGL (GPU-based) instead of looking for a monitor.

### Step 5: Send Commands

In command line mode, type:
```
start walk forward
start perform a golf swing
stop
quit
```

In file mode, create `text.jsonl`:
```json
{"frame": 0, "text": "walk forward"}
{"frame": 200, "text": "turn left"}
{"frame": 400, "text": "wave hands"}
```

### Distributed Setup (Server on Machine A, Client on Machine B)

If your server GPU is on a different machine:

```bash
# On Machine B (client), set up SSH tunnel:
ssh -L 8000:<server_internal_ip>:8000 -p <ssh_port> user@ssh-host

# Then run client normally — it connects to localhost:8000 via tunnel
python robot_client.py --use_commandline
```

---

## Tunable Parameters

### Client Config (`robot_client.py` main function)

| Parameter | Default | What it does | Tuning advice |
|-----------|---------|--------------|---------------|
| `server_host` | `localhost` | Server address | Change if server is remote |
| `server_port` | `8000` | Server TCP port | Must match server |
| `frequency` | `50` | Control loop Hz | Don't change unless you know what you're doing |
| `ready_threshold` | `30` | Tokens needed before starting | Higher = more startup delay, but smoother start |
| `buffer_threshold` | `50` | Request more tokens when queue drops below this | Higher = more buffer, less chance of starvation |
| `keep_tokens_on_new_instruction` | `30` | Tokens kept when switching prompts | Higher = smoother transitions between motions |
| `keep_tokens_for_generate` | `48` | Historical context tokens sent to server | Must match server's decoder overlap. Higher = better context |
| `read_batch_size` | `20` | Tokens requested per server call | Higher = fewer requests but more latency per batch |

### Simulation Flags (`robot_client.py` top of file)

| Flag | Default | What it does |
|------|---------|--------------|
| `RECORD_VIDEO` | `True` | Save frames to `simulation_output.mp4`. Set `True` for AWS headless |
| `VIDEO_FPS` | `50` | Output video framerate |
| `DEBUG` | `True` | Use MuJoCo sim (vs real hardware) |
| `SIM` | `True` | Full physics sim (vs kinematic-only positioning) |
| `ONLINE_MOTION` | `False` | `True` = generate from server, `False` = load from pkl |
| `START_IDX_50FPS` | `5196` | Which frame in the motion library to start from |

### Config File (`configs/g1_ref_real.yaml`)

| Key | What it does |
|-----|--------------|
| `xml_path` | Path to G1 MJCF robot model |
| `policy_path` | Path to RL policy `.pt` file |
| `motion_file` | Path to pre-recorded motion `.pkl` |
| `simulation_duration` | Max sim time in seconds (default 60) |
| `simulation_dt` | Physics timestep (default 0.001 = 1kHz) |
| `control_decimation` | Physics steps per control step (default 20 → 50Hz control) |

### PD Gains (in `G1.__init__()`)

Per-joint P and D gains control how stiff/responsive each joint is:
- **P gains**: spring strength. Higher = snappier tracking, but can oscillate
- **D gains**: damping. Higher = smoother but slower response
- Legs have higher gains (stability critical), arms have lower gains (more compliant)

---

## Commands Reference

### Command Line Mode (`--use_commandline`)

| Command | Action |
|---------|--------|
| `start <prompt>` | Generate motion from text (e.g. `start walk forward`) |
| `stop` | Stop current generation |
| `quit` | Exit program |

### File Mode (`--use_text_file`, default)

Create `text.jsonl` in project root:
```json
{"frame": 0, "text": "walk forward"}
{"frame": 100, "text": "turn left"}
```
- `frame`: at which control step to send the command (50 frames = 1 second)
- `text`: the motion prompt

---

## Key Components

### `fsq.py` — Finite Scalar Quantization
The VQ-VAE discretization scheme. Maps continuous motion features to discrete tokens using levels `[8, 8, 8, 6, 5]` = 15,360 codebook entries. No learned codebook — codes are deterministic from the level grid.

### `infer_robot.py` — LLM Inference Utilities
- `prepare_inference_input_t2m()` — formats text prompt for Qwen with special tokens
- `unified_generation_step()` — one autoregressive step with KV-cache
- `parse_generated_ids()` / `encode_motion_tokens()` — convert between model token IDs and FSQ code indices
- Motion tokens use a special ID range starting at 129,627

### `infer_fsq_ar.py` — FSQ Autoregressive Decoder
- `TokenDecoder` class loads the FSQ-AE model
- `ar_token_decode()` — takes previous codes + new code, decodes to joint-level motion
- Uses a sliding window of 8 previous codes for context

### Motion Library (`motion_lib/`)
- `motion_lib_h1.py` — manages pre-recorded motion database
- `motion_lib_base.py` — base class for motion loading/querying
- `skeleton.py` — skeleton tree from MJCF, joint hierarchy
- `rotation3d.py` + `motion_utils/` — quaternion math, rotation conversions

### `isaac_utils/` + `torch_jit_utils.py`
Rotation/quaternion utilities (quat_mul, quat_rotate, etc.) used throughout.

---

## The Robot: Unitree G1

- **29 joints**: 6 per leg, 3 waist, 7 per arm
- Two joint orderings: **BYD** (training) and **MuJoCo** (sim) — mapped via index tables in `robot_client.py`
- PD gains tuned per joint (legs stiffer, arms softer)

### Observation Vector (151D)

```
ref_joint_pos     [29]  ← from proxy (decoded motion tokens)
ref_joint_vel     [29]
angular_velocity  [ 3]  ← IMU
cur_joint_pos     [29]  ← current - action_offset
cur_joint_vel     [29]
projected_gravity [ 3]  ← gravity in body frame
prev_action       [29]
```

Normalized using training statistics (`obs_mean`, `obs_std`).

---

## Data Files

| File | Purpose |
|------|---------|
| `raw_walking_g1_deploy.pkl` (24MB) | Pre-recorded reference motion database |
| `configs/g1_ref_real.yaml` | Main config (model paths, sim params, motion settings) |
| `unitree_description/mjcf/g1.xml` | G1 robot model for MuJoCo (29 DOF) |
| `t2m/*.npy` (66 files) | Text-to-motion eval data (condition, output tokens, GT tokens) |
| `text.jsonl` | Scheduled text commands (frame → prompt pairs) |

### Output Files

| File | When created |
|------|-------------|
| `simulation_output.mp4` | When `RECORD_VIDEO=True` |
| `motion_records.npy` | Recorded ref motion (dof_pos, dof_vel) |
| `proxy_time_hot.txt` | Proxy request/response latencies |
| `generated_tokens_5.txt` | Server-side generated token log |
| `read_tokens_response_time_5090.txt` | Server-side response time log |

---

## Benchmarking

### `benchmark_t2m.py`

Evaluates the 66 text-to-motion samples in `t2m/`:

```bash
python3 benchmark_t2m.py
```

**Metrics computed:**
- **Token accuracy** — exact match rate between model output and ground truth
- **Code-space smoothness** — velocity and jerk in FSQ code-space (lower jerk = smoother motion)
- **Motion diversity** — intra-sequence (unique tokens per sequence) and cross-sample (centroid distances)
- **Codebook utilization** — unique tokens used, Shannon entropy, KL divergence

**Latest results:**
- 100% token accuracy (66/66 perfect sequences)
- 43.5% codebook utilization (6,675 / 15,360 codes)
- 88.1% normalized entropy
- Smoothest motions: yoga, stationary gestures (jerk ~1.4)
- Jerkiest motions: rapid retreats, boxing evasion (jerk ~5.4)

---

## Flow Summary

```
Text prompt (file / commandline / voice agent RPC)
    ↓
Proxy.send_start_command(prompt)
    ↓
Qwen 2.5 VL (server.py) → FSQ token IDs
    ↓
Proxy (proxy.py) → decode tokens → joint pos/vel
    ↓
Robot Client (robot_client.py) → 151D obs → RL policy → 29D action
    ↓
PD control → MuJoCo physics (or real G1 hardware)
```

---

## Project Structure

```
uniact-code/
├── robot_client.py          # Main controller (50Hz loop, PD control, MuJoCo sim)
├── server.py                # LLM motion generation server (GPU)
├── proxy.py                 # Token buffer/interpolator between server & client
├── livekit_voice_agent.py   # Voice pipeline agent (STT/LLM/TTS → RPC)
├── livekit_bridge.py        # LiveKit ↔ UniAct bridge (perform_motion RPC)
├── livekit_connect.py       # Token generator + agent dispatcher
├── infer_robot.py           # Qwen inference utilities (KV-cache, token parsing)
├── infer_fsq_ar.py          # FSQ autoregressive decoder
├── fsq.py                   # Finite Scalar Quantization implementation
├── benchmark_t2m.py         # Text-to-motion benchmark script
├── configs/
│   └── g1_ref_real.yaml     # Main configuration
├── motion_lib/              # Motion processing library
│   ├── motion_lib_h1.py     # Motion database manager
│   ├── motion_lib_base.py   # Base motion loading
│   ├── skeleton.py          # Skeleton tree from MJCF
│   └── rotation3d.py        # Rotation utilities
├── unitree_description/
│   └── mjcf/g1.xml          # G1 robot model (29 DOF)
├── data/motion_data/
│   └── raw_walking_g1_deploy.pkl
├── t2m/                     # 66 text-to-motion eval samples
├── isaac_utils/             # Rotation/math utilities
├── torch_jit_utils.py       # TorchScript quaternion ops
├── mujoco_utils.py          # MuJoCo helpers
├── .env                     # LiveKit credentials (gitignored — copy from .env.example)
├── .env.example             # Credentials template
├── requirements.txt         # Client dependencies (includes livekit, python-dotenv)
└── requirements_qwen.txt    # Server dependencies (Qwen LLM)
```
