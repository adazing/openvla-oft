"""
resource_test_vla.py

Measures VLA model resource usage:
1. Total model parameters
2. Trainable model parameters
3. Inference speed (control frequency)
"""

import logging
import sys
import time

import hydra
import numpy as np
import torch
from omegaconf import DictConfig

# Append current directory so that interpreter can find experiments.robot
sys.path.append("../..")
from experiments.robot.openvla_utils import (
    get_action_head,
    get_noisy_action_projector,
    get_processor,
    get_proprio_projector,
    resize_image_for_policy,
)
from experiments.robot.robot_utils import (
    get_action,
    get_image_resize_size,
    get_model,
    set_seed_everywhere,
)
from omegaconf import OmegaConf
from prismatic.vla.constants import set_constants


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def count_parameters(model) -> tuple[int, int]:
    """Count total and trainable parameters in a model."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def print_model_stats(
    model,
    action_head=None,
    proprio_projector=None,
    noisy_action_projector=None,
):
    """Print parameter counts for all model components."""
    print("\n" + "=" * 60)
    print("MODEL PARAMETER COUNTS")
    print("=" * 60)

    # Main model
    total, trainable = count_parameters(model)
    print(f"\nMain Model (VLA):")
    print(f"  Total parameters:     {total:,} ({total / 1e9:.2f}B)")
    print(f"  Trainable parameters: {trainable:,} ({trainable / 1e6:.2f}M)")

    grand_total = total
    grand_trainable = trainable

    # Action head
    if action_head is not None:
        total, trainable = count_parameters(action_head)
        print(f"\nAction Head:")
        print(f"  Total parameters:     {total:,} ({total / 1e6:.2f}M)")
        print(f"  Trainable parameters: {trainable:,} ({trainable / 1e6:.2f}M)")
        grand_total += total
        grand_trainable += trainable

    # Proprio projector
    if proprio_projector is not None:
        total, trainable = count_parameters(proprio_projector)
        print(f"\nProprio Projector:")
        print(f"  Total parameters:     {total:,} ({total / 1e6:.2f}M)")
        print(f"  Trainable parameters: {trainable:,} ({trainable / 1e6:.2f}M)")
        grand_total += total
        grand_trainable += trainable

    # Noisy action projector
    if noisy_action_projector is not None:
        total, trainable = count_parameters(noisy_action_projector)
        print(f"\nNoisy Action Projector:")
        print(f"  Total parameters:     {total:,} ({total / 1e6:.2f}M)")
        print(f"  Trainable parameters: {trainable:,} ({trainable / 1e6:.2f}M)")
        grand_total += total
        grand_trainable += trainable

    print(f"\n" + "-" * 60)
    print(f"GRAND TOTAL:")
    print(f"  Total parameters:     {grand_total:,} ({grand_total / 1e9:.2f}B)")
    print(f"  Trainable parameters: {grand_trainable:,} ({grand_trainable / 1e6:.2f}M)")
    print("=" * 60 + "\n")


def create_dummy_observation(cfg: DictConfig, resize_size: int) -> dict:
    """Create a dummy observation for inference testing."""
    # Create a random image of the expected size
    img = np.random.randint(0, 256, (resize_size, resize_size, 3), dtype=np.uint8)
    observation = {"full_image": img}

    # Add wrist image if multi-view
    if cfg.num_images_in_input > 1:
        wrist_img = np.random.randint(0, 256, (resize_size, resize_size, 3), dtype=np.uint8)
        observation["wrist_image"] = wrist_img

    # Add proprio state if used
    if cfg.use_proprio:
        observation["state"] = np.random.randn(cfg.proprio_dim).astype(np.float32)

    return observation


def measure_inference_speed(
    cfg: DictConfig,
    model,
    observation: dict,
    task_description: str,
    processor=None,
    action_head=None,
    proprio_projector=None,
    noisy_action_projector=None,
    num_iterations: int = 100,
    warmup_iterations: int = 10,
):
    """Measure inference speed by running the same observation N times."""
    print("\n" + "=" * 60)
    print("INFERENCE SPEED TEST")
    print("=" * 60)
    print(f"\nWarmup iterations: {warmup_iterations}")
    print(f"Timed iterations:  {num_iterations}")

    # Warmup runs (not timed)
    print("\nRunning warmup...")
    for _ in range(warmup_iterations):
        _ = get_action(
            cfg,
            model,
            observation,
            task_description,
            processor=processor,
            action_head=action_head,
            proprio_projector=proprio_projector,
            noisy_action_projector=noisy_action_projector,
            use_film=cfg.use_film,
        )

    # Synchronize CUDA before timing
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # Timed runs
    print("Running timed iterations...")
    times = []
    for i in range(num_iterations):
        start_time = time.perf_counter()

        _ = get_action(
            cfg,
            model,
            observation,
            task_description,
            processor=processor,
            action_head=action_head,
            proprio_projector=proprio_projector,
            noisy_action_projector=noisy_action_projector,
            use_film=cfg.use_film,
        )

        # Synchronize CUDA after each inference
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        end_time = time.perf_counter()
        times.append(end_time - start_time)

    # Calculate statistics
    times = np.array(times)
    mean_time = np.mean(times)
    std_time = np.std(times)
    min_time = np.min(times)
    max_time = np.max(times)
    median_time = np.median(times)

    # Control frequency
    control_freq = 1.0 / mean_time

    print(f"\n" + "-" * 60)
    print("RESULTS:")
    print(f"  Mean inference time:   {mean_time * 1000:.2f} ms")
    print(f"  Std inference time:    {std_time * 1000:.2f} ms")
    print(f"  Min inference time:    {min_time * 1000:.2f} ms")
    print(f"  Max inference time:    {max_time * 1000:.2f} ms")
    print(f"  Median inference time: {median_time * 1000:.2f} ms")
    print(f"\n  Control Frequency:     {control_freq:.2f} Hz")
    print("=" * 60 + "\n")

    return {
        "mean_time": mean_time,
        "std_time": std_time,
        "min_time": min_time,
        "max_time": max_time,
        "median_time": median_time,
        "control_freq": control_freq,
    }


def check_unnorm_key(cfg: DictConfig, model) -> None:
    """Check that the model contains the action un-normalization key."""
    unnorm_key = cfg.unnorm_key

    if not unnorm_key:
        unnorm_key = cfg.env_name

    if unnorm_key not in model.norm_stats:
        candidates = [unnorm_key, unnorm_key.replace("_", "")]
        if len(model.norm_stats) == 1:
            unnorm_key = next(iter(model.norm_stats.keys()))
        else:
            found = False
            for candidate in candidates:
                if candidate in model.norm_stats:
                    unnorm_key = candidate
                    found = True
                    break
            if not found:
                raise ValueError(
                    f"Action un-norm key '{unnorm_key}' not found in VLA norm_stats. "
                    f"Available keys: {list(model.norm_stats.keys())}. "
                    f"Set unnorm_key= to one of these."
                )

    OmegaConf.set_struct(cfg, False)
    cfg.unnorm_key = unnorm_key


def initialize_model(cfg: DictConfig):
    """Initialize model and associated components."""
    model = get_model(cfg)

    proprio_projector = None
    if cfg.use_proprio:
        proprio_projector = get_proprio_projector(
            cfg,
            model.llm_dim,
            proprio_dim=cfg.proprio_dim,
        )

    action_head = None
    if cfg.use_l1_regression or cfg.use_diffusion:
        action_head = get_action_head(cfg, model.llm_dim)

    noisy_action_projector = None
    if cfg.use_diffusion:
        noisy_action_projector = get_noisy_action_projector(cfg, model.llm_dim)

    processor = None
    if cfg.model_family == "openvla":
        processor = get_processor(cfg)

    return model, action_head, proprio_projector, noisy_action_projector, processor


@hydra.main(version_base="1.2", config_path="configs", config_name="eval")
def run_resource_test(cfg: DictConfig) -> None:
    """Main function to test VLA model resources and inference speed."""
    # Set robot constants from config
    set_constants(
        action_dim=cfg.action_dim,
        num_actions_chunk=cfg.num_actions_chunk,
        proprio_dim=cfg.proprio_dim,
        normalization_type=cfg.normalization_type,
    )

    # Set random seed
    set_seed_everywhere(cfg.seed)

    print("\n" + "=" * 60)
    print("VLA RESOURCE TEST")
    print("=" * 60)
    print(f"Model family: {cfg.model_family}")
    print(f"Checkpoint: {cfg.pretrained_checkpoint}")
    print(f"Use diffusion: {cfg.use_diffusion}")
    print(f"Use L1 regression: {cfg.use_l1_regression}")
    print(f"Use proprio: {cfg.use_proprio}")

    # Initialize model and components
    print("\nLoading model...")
    model, action_head, proprio_projector, noisy_action_projector, processor = initialize_model(cfg)

    # Check unnorm_key for OpenVLA models
    if cfg.model_family == "openvla":
        check_unnorm_key(cfg, model)

    # Print parameter counts
    print_model_stats(model, action_head, proprio_projector, noisy_action_projector)

    # Get expected image dimensions
    resize_size = get_image_resize_size(cfg)

    # Create dummy observation
    observation = create_dummy_observation(cfg, resize_size)
    task_description = "test task description"

    # Measure inference speed
    num_iterations = cfg.get("num_inference_iterations", 100)
    warmup_iterations = cfg.get("warmup_iterations", 10)

    measure_inference_speed(
        cfg,
        model,
        observation,
        task_description,
        processor=processor,
        action_head=action_head,
        proprio_projector=proprio_projector,
        noisy_action_projector=noisy_action_projector,
        num_iterations=num_iterations,
        warmup_iterations=warmup_iterations,
    )


if __name__ == "__main__":
    run_resource_test()
