# Repository Guidelines

## Project Structure & Module Organization
- `toolkit/` — core training logic (models, pipelines, samplers, adapters).
- `config/` — runnable YAML configs (`config/examples/*.yaml`), keymaps, and mapping helpers.
- `extensions/` + `extensions_built_in/` — optional add-ons; keep custom code here to avoid touching core.
- `scripts/` — utilities for converting models, repairing datasets, generating sampler scales.
- `testing/` — manual evaluation scripts (bucket loader, VAE metrics, key diffing).
- `ui/` — Next.js 15 + Prisma dashboard with worker in `cron/`.
- `docker/` + `docker-compose.yml` — container entrypoints.

## Build, Test, and Development Commands
```bash
# Python backend
python3 -m venv venv && source venv/bin/activate
pip install --no-cache-dir torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
python run.py config/examples/train_lora_flex_24gb.yaml -n my_run

# UI
cd ui && npm install
npm run dev            # hot reload UI + worker
npm run build && npm run start   # production build/serve on :8675
npm run lint | npm run format
npm run update_db      # regenerate Prisma client & push schema

# Targeted checks (GPU optional)
python testing/test_bucket_dataloader.py <dataset_dir> --epochs 1
python testing/test_vae.py --help
```

## Coding Style & Naming Conventions
- Python: PEP 8, 4-space indent, snake_case for functions/vars, PascalCase for classes; use `toolkit.print.print_acc` for accelerator-aware logging.
- Configs: descriptive snake_case (`train_<model>_<variant>.yaml`); keep custom assets in `config/` or `assets/`.
- UI: TypeScript + functional components; colocate styles; rely on Prettier (`npm run format`).

## Testing Guidelines
- Add checks under `testing/` with explicit CLI args; fail fast on missing data/weights.
- Gate GPU-heavy paths with env flags (e.g., `RUN_SLOW=1`) so CPU smoke tests still run.
- UI changes: run `npm run lint`; add screenshots when altering shared components.

## Commit & Pull Request Guidelines
- Follow repo history: short, imperative subjects (e.g., “Fix text encoder unload”); no trailing punctuation.
- PR checklist: summary of behavior change, configs or schemas touched, test commands executed (backend + UI), and any new asset/weight locations.
- UI/UX updates need before/after screenshots; backend changes should include a sample training invocation/output.
- Link related issues; request review for core (`toolkit/`) or database schema edits.

## Security & Configuration Notes
- Keep secrets in a local `.env`; `run.py` auto-loads it. Never commit private weights/datasets.
- Training assumes CUDA 12.6-class GPUs; add CPU fallbacks or guards when introducing new ops.
- Mirror pinned versions above when updating Docker or CI.
