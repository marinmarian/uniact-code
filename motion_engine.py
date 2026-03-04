"""
In-process motion generator + FSQ decoder.

Replaces server.py (MotionServer) and proxy.py (MotionProxy buffering/interpolation)
with a socket-free async class that can be used directly by unified_agent.py.
"""

import asyncio
import time
from collections import deque

import numpy as np
import torch

from fsq import FSQ
from infer_robot import (
    MOTION_TOKEN_CONFIG,
    load_finetuned_model,
    unified_generation_step,
)

# ---------------------------------------------------------------------------
# Joint ordering (copied from server.py — must stay in sync)
# ---------------------------------------------------------------------------

BYD_JOINT_NAMES = [
    "left_hip_pitch_joint", "right_hip_pitch_joint", "waist_yaw_joint",
    "left_hip_roll_joint", "right_hip_roll_joint", "waist_roll_joint",
    "left_hip_yaw_joint", "right_hip_yaw_joint", "waist_pitch_joint",
    "left_knee_joint", "right_knee_joint",
    "left_shoulder_pitch_joint", "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint", "right_ankle_pitch_joint",
    "left_shoulder_roll_joint", "right_shoulder_roll_joint",
    "left_ankle_roll_joint", "right_ankle_roll_joint",
    "left_shoulder_yaw_joint", "right_shoulder_yaw_joint",
    "left_elbow_joint", "right_elbow_joint",
    "left_wrist_roll_joint", "right_wrist_roll_joint",
    "left_wrist_pitch_joint", "right_wrist_pitch_joint",
    "left_wrist_yaw_joint", "right_wrist_yaw_joint",
]

MUJOCO_JOINT_NAMES = [
    "left_hip_pitch", "left_hip_roll", "left_hip_yaw", "left_knee",
    "left_ankle_pitch", "left_ankle_roll",
    "right_hip_pitch", "right_hip_roll", "right_hip_yaw", "right_knee",
    "right_ankle_pitch", "right_ankle_roll",
    "waist_yaw", "waist_roll", "waist_pitch",
    "left_shoulder_pitch", "left_shoulder_roll", "left_shoulder_yaw",
    "left_elbow", "left_wrist_roll", "left_wrist_pitch", "left_wrist_yaw",
    "right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw",
    "right_elbow", "right_wrist_roll", "right_wrist_pitch", "right_wrist_yaw",
]

BYD_TO_MUJOCO = [BYD_JOINT_NAMES.index(name + "_joint") for name in MUJOCO_JOINT_NAMES]


class MotionEngine:
    """In-process Qwen motion token generator + FSQ decoder.

    All GPU work is run via ``run_in_executor`` so the LiveKit async event
    loop is never blocked.  An ``asyncio.Lock`` (``gpu_lock``) serialises
    GPU access with the chat LLM.
    """

    def __init__(
        self,
        model_path: str,
        decoder_path: str,
        gpu_lock: asyncio.Lock | None = None,
    ):
        self.gpu_lock = gpu_lock or asyncio.Lock()
        self._gen_task: asyncio.Task | None = None

        # Decoded motion frame queue (consumed by the 50Hz streamer)
        self._frame_queue: deque[dict] = deque()
        self._last_frame: dict = {"dof_pos": [0.0] * 29, "dof_vel": [0.0] * 29}

        # Token cache (mirrors server.py's token_cache / dict_cache logic)
        self._token_cache: deque[int] = deque()
        self._dict_cache_size = 0  # tracks how many tokens have been decoded

        # ---- Load Qwen model ----
        print(f"[MotionEngine] Loading model from {model_path} ...")
        self._model, self._tokenizer = load_finetuned_model(model_path)
        print("[MotionEngine] Model loaded.")

        # ---- Decoder + FSQ quantizer ----
        self._decoder = torch.jit.load(decoder_path)
        self._decoder.eval()
        self._quantize = FSQ(levels=[8, 8, 8, 6, 5])
        self._quantize = self._quantize.to(torch.cuda.current_device())

        # Denormalization tensors (from server.py)
        self._min_vals = torch.tensor(
            [-1.5348, -1.5571, -0.5521, -0.2563, -0.6761, -0.4234, -0.4155,
             -0.6174, -0.3280,  0.0206,  0.0459, -2.5961, -2.7955, -0.7617,
             -0.7957, -0.4254, -2.2515, -0.2618, -0.2551, -1.4000, -1.9968,
             -0.9473, -1.0472, -1.5193, -0.8290, -0.5298, -1.3960, -1.5992,
             -1.6144],
            device="cuda:0",
        )
        self._value_range = torch.tensor(
            [1.7558, 1.7926, 1.1905, 0.9004, 0.9268, 0.8572, 1.0855, 1.0616,
             0.8480, 1.7819, 1.8609, 3.7451, 3.9445, 1.2204, 1.1841, 2.6769,
             2.7070, 0.5173, 0.5169, 3.2215, 3.3968, 2.6473, 2.7151, 2.3118,
             2.8012, 1.4724, 3.0105, 3.2137, 3.2289],
            device="cuda:0",
        )

        # Warm-up decoder (following server.py init_decoder)
        self._warmup_decoder()
        self._warmup_generation()

    # ------------------------------------------------------------------
    # Properties exposed for QwenLLM to share the model
    # ------------------------------------------------------------------

    @property
    def model(self):
        return self._model

    @property
    def tokenizer(self):
        return self._tokenizer

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def start_generation(self, prompt: str) -> None:
        """Cancel any running generation and start a new one for *prompt*."""
        if self._gen_task and not self._gen_task.done():
            self._gen_task.cancel()
            try:
                await self._gen_task
            except asyncio.CancelledError:
                pass

        # Clear caches
        self._token_cache.clear()
        self._dict_cache_size = 0
        self._frame_queue.clear()

        self._gen_task = asyncio.get_event_loop().create_task(
            self._generate_loop(prompt)
        )

    def read_motion_frame(self) -> dict | None:
        """Pop one decoded motion frame, or None if empty."""
        if self._frame_queue:
            return self._frame_queue.popleft()
        return None

    def queue_size(self) -> int:
        return len(self._frame_queue)

    # ------------------------------------------------------------------
    # Generation loop (runs as asyncio Task)
    # ------------------------------------------------------------------

    async def _generate_loop(self, prompt: str) -> None:
        """Autoregressive token generation + periodic batch decode."""
        loop = asyncio.get_event_loop()
        try:
            # First step: process prompt
            async with self.gpu_lock:
                next_token_id, past_kv, prompt_length = await loop.run_in_executor(
                    None, self._first_step, prompt
                )

            step = 0
            while True:
                # Add token to cache
                token_item = next_token_id.item() - MOTION_TOKEN_CONFIG["code_base_id"]
                self._token_cache.append(token_item)

                # Batch decode when we have 32 undecoded tokens
                undecoded = len(self._token_cache) - self._dict_cache_size
                if undecoded >= 32:
                    async with self.gpu_lock:
                        await loop.run_in_executor(None, self._batch_decode)

                # Generate next token
                async with self.gpu_lock:
                    next_token_id, past_kv, prompt_length = await loop.run_in_executor(
                        None,
                        self._next_step,
                        next_token_id,
                        past_kv,
                        prompt_length,
                        step,
                    )
                step += 1

                # Yield control briefly so other tasks (streamer, chat) can run
                await asyncio.sleep(0)

        except asyncio.CancelledError:
            # Flush any remaining undecoded tokens
            undecoded = len(self._token_cache) - self._dict_cache_size
            if undecoded > 0:
                try:
                    async with self.gpu_lock:
                        await asyncio.get_event_loop().run_in_executor(
                            None, self._batch_decode
                        )
                except Exception:
                    pass
        except Exception as exc:
            print(f"[MotionEngine] Generation error: {exc}")

    # ------------------------------------------------------------------
    # GPU work (called via run_in_executor, no async)
    # ------------------------------------------------------------------

    def _first_step(self, prompt: str):
        """Prompt processing — returns (next_token_id, past_kv, prompt_length)."""
        prompt_length = 0
        next_token_id, past_kv, _is_first, _is_end, prompt_length = (
            unified_generation_step(
                self._model,
                self._tokenizer,
                prompt=prompt,
                prompt_length=prompt_length,
                motion_tokens=None,
                past_key_values=None,
                step_count=0,
            )
        )
        return next_token_id, past_kv, prompt_length

    def _next_step(self, prev_token_id, past_kv, prompt_length, step):
        """Single autoregressive step."""
        next_token_id, past_kv, _is_first, _is_end, prompt_length = (
            unified_generation_step(
                self._model,
                self._tokenizer,
                prompt=None,
                prompt_length=prompt_length,
                motion_tokens=prev_token_id,
                past_key_values=past_kv,
                step_count=step + 1,
            )
        )
        return next_token_id, past_kv, prompt_length

    def _batch_decode(self) -> None:
        """Decode all undecoded tokens in ``_token_cache`` to motion frames.

        Mirrors ``MotionServer.get_token_dict()`` — converts FSQ indices to
        codes, runs decoder, denormalises, reorders joints, computes velocity.
        """
        undecoded = len(self._token_cache) - self._dict_cache_size
        if undecoded <= 0:
            return

        # Gather undecoded tokens (they sit at the tail of _token_cache)
        all_tokens = list(self._token_cache)
        tokens_to_process = all_tokens[self._dict_cache_size:]

        gen_token_ids = torch.tensor(tokens_to_process, dtype=torch.int64, device="cuda")

        gen_codes = self._quantize.indices_to_codes(gen_token_ids)
        with torch.no_grad():
            gen_codes = gen_codes.unsqueeze(0).cuda()
            output = self._decoder(gen_codes)

        # Denormalize
        output = self._denormalize(output)

        # output shape: (1, undecoded*2, 29)
        # Each token produces 2 frames (pos interleaved at 50Hz)
        for i in range(output.shape[1]):
            frame_tensor = output[:, i, :]
            frame_tensor = frame_tensor[:, BYD_TO_MUJOCO]

            prev_pos = torch.tensor(
                self._last_frame["dof_pos"], device=frame_tensor.device
            )
            dof_vel = (frame_tensor - prev_pos) * 50

            dof_pos_list = frame_tensor.cpu().numpy().tolist()
            dof_vel_list = dof_vel.cpu().numpy().tolist()
            frame = {"dof_pos": dof_pos_list, "dof_vel": dof_vel_list}
            self._last_frame = frame
            self._frame_queue.append(frame)

        # Mark all tokens as decoded (each token → 2 dict entries, but we
        # track at the token level; the duplicate-append from server.py is
        # handled implicitly because the decoder outputs 2× frames per token).
        self._dict_cache_size = len(self._token_cache)

    def _denormalize(self, data: torch.Tensor) -> torch.Tensor:
        """Denormalize decoder output (from server.py)."""
        data = (data + 1).cuda()
        return data * self._value_range / 2 + self._min_vals

    # ------------------------------------------------------------------
    # Interpolation (from proxy.py)
    # ------------------------------------------------------------------

    @staticmethod
    def interpolate_frames(
        frame_start: dict, frame_end: dict, num_points: int = 20
    ) -> list[dict]:
        """Linear interpolation between two motion frames.

        Returns *num_points* intermediate frames (excluding start and end).
        """
        pos_start = np.array(frame_start["dof_pos"])
        pos_end = np.array(frame_end["dof_pos"])
        result = []
        prev_pos = pos_start
        for i in range(1, num_points + 1):
            alpha = i / (num_points + 1)
            interp_pos = pos_start * (1 - alpha) + pos_end * alpha
            vel = (interp_pos - prev_pos) * 50
            result.append({"dof_pos": interp_pos.tolist(), "dof_vel": vel.tolist()})
            prev_pos = interp_pos
        return result

    # ------------------------------------------------------------------
    # Warm-up helpers
    # ------------------------------------------------------------------

    def _warmup_decoder(self) -> None:
        """Run decoder on random tokens to trigger JIT compilation."""
        token_ids = torch.randint(1, 100, (1, 32), device=torch.cuda.current_device())
        gen_codes = self._quantize.indices_to_codes(token_ids.squeeze(0))
        with torch.no_grad():
            gen_codes = gen_codes.unsqueeze(0).cuda()
            output = self._decoder(gen_codes)
        self._denormalize(output)
        print("[MotionEngine] Decoder warm-up done.")

    def _warmup_generation(self) -> None:
        """Run a few generation steps to warm up the LLM."""
        prompt_length = 0
        next_token_id, past_kv, _, _, prompt_length = unified_generation_step(
            self._model,
            self._tokenizer,
            prompt="hello qwen",
            prompt_length=prompt_length,
            motion_tokens=None,
            past_key_values=None,
            step_count=0,
        )
        for step in range(3):
            next_token_id, past_kv, _, _, prompt_length = unified_generation_step(
                self._model,
                self._tokenizer,
                prompt=None,
                prompt_length=prompt_length,
                motion_tokens=next_token_id,
                past_key_values=past_kv,
                step_count=step + 1,
            )
        print("[MotionEngine] Generation warm-up done.")
