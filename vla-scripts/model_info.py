"""
model_info.py

Prints total parameters, trainable parameters, and measures inference speed (Hz) for OpenVLA-OFT.

Supports the full OFT component stack: base VLA (with LoRA), action head (L1 or diffusion),
proprioception projector, and optionally the noisy action projector (diffusion only).

Usage:
    python vla-scripts/model_info.py
    python vla-scripts/model_info.py --vla_path openvla/openvla-7b
    python vla-scripts/model_info.py --vla_path <PATH/TO/LOCAL/CHECKPOINT>
    python vla-scripts/model_info.py --use_diffusion --num_diffusion_steps_train 50 --num_diffusion_steps_inference 10
    python vla-scripts/model_info.py --robot_platform ALOHA
    python vla-scripts/model_info.py --num_actions_chunk 1  # disable action chunking for speed comparison
"""

import argparse
import time

import numpy as np
import torch
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor

from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
from prismatic.models.action_heads import DiffusionActionHead, L1RegressionActionHead
from prismatic.models.projectors import NoisyActionProjector, ProprioProjector
import prismatic.vla.constants as C


def fmt_params(n):
    """Format parameter count with appropriate unit."""
    if n >= 1e9:
        return f"{n:>14,}  ({n / 1e9:.2f}B)"
    else:
        return f"{n:>14,}  ({n / 1e6:.2f}M)"


def count_params(module):
    """Return (total, trainable) parameter counts for a module."""
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return total, trainable


def main():
    parser = argparse.ArgumentParser(description="Print OpenVLA-OFT model info: params & inference speed")
    parser.add_argument("--vla_path", type=str, default="openvla/openvla-7b", help="Path to base OpenVLA model")
    parser.add_argument("--lora_rank", type=int, default=32, help="LoRA rank for fine-tuning")
    parser.add_argument("--no_lora", action="store_true", help="Skip LoRA (show base model only)")

    # Action head (L1 regression by default; pass --use_diffusion to switch)
    parser.add_argument("--use_diffusion", action="store_true", help="Use diffusion action head instead of L1 regression")
    parser.add_argument("--num_diffusion_steps_train", type=int, default=50, help="Diffusion training steps")
    parser.add_argument("--num_diffusion_steps_inference", type=int, default=10, help="Diffusion inference steps")

    # Proprio (included by default; pass --no_proprio to skip)
    parser.add_argument("--no_proprio", action="store_true", help="Skip proprio projector")

    # Robot platform (sets ACTION_DIM, PROPRIO_DIM, NUM_ACTIONS_CHUNK)
    parser.add_argument("--robot_platform", type=str, default="LIBERO", choices=["LIBERO", "ALOHA", "BRIDGE"],
                        help="Robot platform for action/proprio dimensions")
    parser.add_argument("--num_actions_chunk", type=int, default=None,
                        help="Override NUM_ACTIONS_CHUNK (e.g. 1 to disable action chunking)")

    # Inference speed
    parser.add_argument("--n_warmup", type=int, default=5, help="Number of warmup inference passes")
    parser.add_argument("--n_trials", type=int, default=20, help="Number of timed inference passes")
    parser.add_argument("--skip_speed", action="store_true", help="Skip inference speed measurement")

    args = parser.parse_args()

    # Derive flags
    args.use_l1_regression = not args.use_diffusion
    args.use_proprio = not args.no_proprio

    # Set robot platform constants
    platform_constants = {
        "LIBERO": C.LIBERO_CONSTANTS,
        "ALOHA": C.ALOHA_CONSTANTS,
        "BRIDGE": C.BRIDGE_CONSTANTS,
    }[args.robot_platform]
    num_actions_chunk = args.num_actions_chunk if args.num_actions_chunk is not None else platform_constants["NUM_ACTIONS_CHUNK"]
    C.set_constants(
        action_dim=platform_constants["ACTION_DIM"],
        num_actions_chunk=num_actions_chunk,
        proprio_dim=platform_constants["PROPRIO_DIM"],
        normalization_type=platform_constants["ACTION_PROPRIO_NORMALIZATION_TYPE"],
    )

    # Register OpenVLA model to HF Auto Classes
    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

    print(f"Loading model from: {args.vla_path}")
    processor = AutoProcessor.from_pretrained(args.vla_path, trust_remote_code=True)
    vla = AutoModelForVision2Seq.from_pretrained(
        args.vla_path,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    vla = vla.to(device)

    # === Parameter Counts ===
    vla_total, vla_trainable = count_params(vla)
    llm_dim = vla.llm_dim

    print("\n" + "=" * 70)
    print("  OpenVLA-OFT Model Info")
    print("=" * 70)
    print(f"  Robot platform:    {args.robot_platform}")
    print(f"  ACTION_DIM:        {C.ACTION_DIM}")
    print(f"  NUM_ACTIONS_CHUNK: {C.NUM_ACTIONS_CHUNK}")
    print(f"  PROPRIO_DIM:       {C.PROPRIO_DIM}")
    print(f"  LLM hidden dim:    {llm_dim}")
    print()

    # --- Base VLA ---
    print("  [Base VLA]")
    print(f"  Total params:      {fmt_params(vla_total)}")
    print(f"  Trainable params:  {fmt_params(vla_trainable)}")

    # --- LoRA ---
    lora_vla = None
    if not args.no_lora:
        from peft import LoraConfig, get_peft_model

        lora_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=min(args.lora_rank, 16),
            lora_dropout=0.0,
            target_modules="all-linear",
            init_lora_weights="gaussian",
        )
        lora_vla = get_peft_model(vla, lora_config)
        lora_total, lora_trainable = count_params(lora_vla)
        print(f"\n  [LoRA r={args.lora_rank}]")
        print(f"  LoRA total:        {fmt_params(lora_total)}")
        print(f"  LoRA trainable:    {fmt_params(lora_trainable)}")
        print(f"  % trainable:       {100 * lora_trainable / lora_total:>13.2f}%")

    # --- Action Head ---
    if args.use_l1_regression:
        action_head = L1RegressionActionHead(
            input_dim=llm_dim, hidden_dim=llm_dim, action_dim=C.ACTION_DIM,
        ).to(device, dtype=torch.float32)
        head_type = "L1 Regression"
    else:
        action_head = DiffusionActionHead(
            input_dim=llm_dim, hidden_dim=llm_dim, action_dim=C.ACTION_DIM,
            num_diffusion_steps_train=args.num_diffusion_steps_train,
        ).to(device, dtype=torch.float32)
        action_head.noise_scheduler.set_timesteps(args.num_diffusion_steps_inference)
        head_type = f"Diffusion (train={args.num_diffusion_steps_train}, infer={args.num_diffusion_steps_inference})"

    ah_total, ah_trainable = count_params(action_head)
    print(f"\n  [Action Head: {head_type}]")
    print(f"  Total params:      {fmt_params(ah_total)}")
    print(f"  Trainable params:  {fmt_params(ah_trainable)}")

    # --- Proprio Projector ---
    proprio_projector = None
    if args.use_proprio:
        proprio_projector = ProprioProjector(
            llm_dim=llm_dim, proprio_dim=C.PROPRIO_DIM,
        ).to(device, dtype=torch.float32)
        pp_total, pp_trainable = count_params(proprio_projector)
        print(f"\n  [Proprio Projector]")
        print(f"  Total params:      {fmt_params(pp_total)}")
        print(f"  Trainable params:  {fmt_params(pp_trainable)}")

    # --- Noisy Action Projector (diffusion only) ---
    noisy_action_projector = None
    if args.use_diffusion:
        noisy_action_projector = NoisyActionProjector(
            llm_dim=llm_dim,
        ).to(device, dtype=torch.float32)
        nap_total, nap_trainable = count_params(noisy_action_projector)
        print(f"\n  [Noisy Action Projector]")
        print(f"  Total params:      {fmt_params(nap_total)}")
        print(f"  Trainable params:  {fmt_params(nap_trainable)}")

    # --- Grand Total ---
    print("\n" + "-" * 70)
    if not args.no_lora:
        grand_total = lora_total + ah_total
        grand_trainable = lora_trainable + ah_trainable
    else:
        grand_total = vla_total + ah_total
        grand_trainable = vla_trainable + ah_trainable
    if proprio_projector is not None:
        grand_total += pp_total
        grand_trainable += pp_trainable
    if noisy_action_projector is not None:
        grand_total += nap_total
        grand_trainable += nap_trainable

    print(f"  [OFT Grand Total]")
    print(f"  All params:        {fmt_params(grand_total)}")
    print(f"  All trainable:     {fmt_params(grand_trainable)}")
    print(f"  % trainable:       {100 * grand_trainable / grand_total:>13.2f}%")

    # === Inference Speed ===
    if not args.skip_speed:
        print("\n" + "-" * 70)
        print(f"  Measuring inference speed on {device} ({args.n_trials} trials)...")

        from PIL import Image

        dummy_image = Image.new("RGB", (224, 224), color=(128, 128, 128))
        prompt = "In: What action should the robot take to pick up the object?\nOut:"
        inputs = processor(prompt, dummy_image).to(device, dtype=torch.float32)

        unnorm_key = next(iter(vla.norm_stats.keys()))

        # After get_peft_model, vla's layers are modified in-place with LoRA adapters,
        # so using vla directly gives realistic inference with LoRA included.
        vla.eval()
        action_head.eval()
        if proprio_projector is not None:
            proprio_projector.eval()
        if noisy_action_projector is not None:
            noisy_action_projector.eval()

        # Create dummy proprio input (numpy array — predict_action converts via torch.Tensor())
        dummy_proprio = np.zeros(C.PROPRIO_DIM, dtype=np.float32) if args.use_proprio else None

        with torch.no_grad():
            # Warmup
            for _ in range(args.n_warmup):
                vla.predict_action(
                    **inputs,
                    unnorm_key=unnorm_key,
                    proprio=dummy_proprio,
                    proprio_projector=proprio_projector,
                    noisy_action_projector=noisy_action_projector,
                    action_head=action_head,
                )

            if device == "cuda":
                torch.cuda.synchronize()

            # Timed trials
            start = time.perf_counter()
            for _ in range(args.n_trials):
                vla.predict_action(
                    **inputs,
                    unnorm_key=unnorm_key,
                    proprio=dummy_proprio,
                    proprio_projector=proprio_projector,
                    noisy_action_projector=noisy_action_projector,
                    action_head=action_head,
                )
            if device == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start

        avg_time = elapsed / args.n_trials
        raw_hz = 1.0 / avg_time
        effective_hz = C.NUM_ACTIONS_CHUNK / avg_time

        print(f"  Avg inference time: {avg_time * 1000:.1f} ms")
        print(f"  Raw forward pass:   {raw_hz:.2f} Hz")
        print(f"  Effective speed:    {effective_hz:.2f} Hz  (×{C.NUM_ACTIONS_CHUNK} action chunks)")

    print("=" * 70)


if __name__ == "__main__":
    main()