# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AI Toolkit is a training suite for diffusion models (FLUX, Stable Diffusion, WAN, etc.) supporting LoRA, LoKr, full fine-tuning, and various adapters. It runs as both CLI and web UI.

## Common Commands

```bash
# Install dependencies
python3 -m venv venv
source venv/bin/activate  # Windows: .\venv\Scripts\activate
pip3 install --no-cache-dir torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu126
pip3 install -r requirements.txt

# Run a training job
python run.py config/your_config.yaml

# Run with recovery (continue on failure)
python run.py -r config/config1.yaml config/config2.yaml

# Run with custom name (replaces [name] in config)
python run.py -n my_experiment config/template.yaml

# Run the web UI (requires Node.js 18+)
cd ui && npm run build_and_start

# Run Gradio UI for simple training
python flux_train_ui.py
```

## Architecture

### Entry Points
- `run.py` - Main CLI entry point. Loads config, dispatches to appropriate Job class
- `flux_train_ui.py` - Gradio-based training UI
- `ui/` - Next.js web UI (runs on port 8675)

### Core Flow
1. `toolkit/job.py:get_job()` reads YAML config and returns appropriate Job instance
2. Job types in `jobs/`: ExtensionJob (most training), TrainJob, GenerateJob, ExtractJob, ModJob
3. Jobs load Process classes that do actual work (e.g., `BaseSDTrainProcess`)

### Extension System
- Extensions live in `extensions/` (custom) and `extensions_built_in/` (shipped)
- Each extension exports `AI_TOOLKIT_EXTENSIONS` list of Extension classes
- Extensions define a `uid` (used in config files) and `get_process()` method
- Key built-in extensions:
  - `sd_trainer` - Main training extension for LoRA/fine-tuning
  - `diffusion_trainer` - Universal trainer for UI/API

### Key Directories
- `toolkit/` - Core library code
  - `stable_diffusion_model.py` - Main model wrapper (~145KB, handles all SD variants)
  - `config_modules.py` - Pydantic config models
  - `data_loader.py`, `dataloader_mixins.py` - Dataset loading with bucketing
  - `models/` - Model implementations (FLUX, LoRA, adapters)
  - `kohya_lora.py`, `lycoris_utils.py` - LoRA/LyCORIS implementations
- `jobs/process/` - Process implementations
  - `BaseSDTrainProcess.py` - Main training process (~115KB)
- `config/examples/` - Example training configs (start here for new configs)

### Config Structure
Configs are YAML with this structure:
```yaml
job: extension  # or train, generate, extract, mod
config:
  name: "my_training_run"
  process:
    - type: 'sd_trainer'  # matches extension uid
      # ... process-specific config
meta:
  name: "[name]"
```

### Supported Models
FLUX.1-dev/schnell, Stable Diffusion 1.5/2.x/XL/3.5, WAN 2.1/2.2, Chroma, OmniGen2, HiDream, Lumina, and more. Check `config/examples/` for model-specific configs.

### Dataset Format
- Folder of images (jpg, jpeg, png) with matching `.txt` caption files
- Images auto-resize and bucket by aspect ratio
- Use `[trigger]` in captions to auto-replace with trigger word
- Supports masked loss training via `mask_path` config
