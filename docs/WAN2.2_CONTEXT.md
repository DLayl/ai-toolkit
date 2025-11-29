# WAN 2.2 Training Documentation

## Overview
This document outlines the architecture, configuration, and training process for WAN 2.2 (14B) models within the `ai-toolkit`. WAN 2.2 utilizes a **Dual Transformer** architecture, separating the diffusion process into "high-noise" and "low-noise" stages, handled by distinct transformer blocks.

## Model Architecture (`wan22_14b`)
- **Location**: `extensions_built_in/diffusion_models/wan22/wan22_14b_model.py`
- **Key Class**: `DualWanTransformer3DModel`
- **Mechanism**: 
  - The model switches between two internal transformers based on the timestep.
  - Configurable via `train_high_noise` and `train_low_noise` flags in the `model` config.

## Configuration Guide

### 1. Base Setup
Training is configured via YAML files. The `arch` must be set to `wan22_14b`.

```yaml
model:
  arch: "wan22_14b"
  name_or_path: "Wan-AI/Wan2.1-T2V-14B" # Or specific 2.2 path if available
  dtype: bf16
  quantize: false # Recommended false for B200/H100
```

### 2. Masked Loss Training
Masked loss allows training only on specific regions of an image (e.g., a specific person or object), ignoring the background.

- **Requirement**: A corresponding black-and-white mask image for every training image.
  - **White (255)**: Area to train (Loss applied).
  - **Black (0)**: Area to ignore (No loss).
- **Configuration**:
  Add `mask_path` to your dataset config.

```yaml
datasets:
  - folder_path: "/path/to/images"
    mask_path: "/path/to/masks" # Must match filenames in folder_path
    caption_ext: "txt"
```
- **Mechanism**: 
  - `MaskFileItemDTOMixin` loads the mask.
  - The trainer converts this to a weight map.
  - MSE Loss is multiplied by this map: `loss = (diff ** 2) * mask_multiplier`.

### 3. Hardware Optimizations (B200/H100)
For high-VRAM cards (80GB+), you can avoid quantization for better quality.

- **Batch Size**: Can be increased (e.g., 4-8 on B200).
- **Quantization**: Set `quantize: false`.
- **Gradient Checkpointing**: Keep `true` for 14B models to prevent OOM on activations, even with high VRAM.

## Data Loading
- **Code**: `toolkit/data_loader.py`
- **Images**: Loaded and resized to `resolution` (e.g., 1024).
- **Videos**: Set `num_frames: N` (where N > 1) to treat inputs as video clips.
