# llm-quant-compare

GGUF quantization drift comparer. Single-script Python tool, no build system, no tests.

## Setup

- Python 3.10+ required
- `python -m pip install -r requirements.txt` (numpy, tqdm, huggingface_hub)

## Run

```
python compare_gguf_quant.py --reference <ref.gguf> --quant <quant.gguf> --out-dir <dir>
```

Outputs go to a timestamped subfolder under `--out-dir` (never overwritten).

Split GGUFs: pass multiple paths or a glob:
```
--reference "./models/BF16/model-*.gguf" --quant "./models/Q8_0/model.gguf"
```

## Architecture

- `compare_gguf_quant.py` is the entire application — parses GGUF binary headers directly, memory-maps tensor data, dequantizes Q8_0 blocks as `scale * int8`, computes per-tensor/layer/sublayer drift metrics.
- No external GGUF library; the GGUF parser lives in this single file.
- Supported tensor types for comparison: BF16, F32, F16, F64, I8/I16/I32/I64, Q8_0.
- Uses `mmap` — multi-GB GGUFs are never fully loaded into RAM.
- Parallel mode (`--workers N`, 0 = auto) uses `ProcessPoolExecutor` with per-worker mmap handles.

## Output files

Each run writes: `report.md`, `tensor_metrics.csv`, `layer_metrics.csv`, `sublayer_metrics.csv`, `metrics.json`.

## Key constraints

- Q8_0 reads must be 32-value block-aligned.
- `.gitignore` excludes `*.gguf`, `models/`, `quant_compare_report/*/`, `.venv/`.
- No tests, lint, typecheck, or CI configured.
- `tqdm` is optional at runtime (graceful fallback if missing).
