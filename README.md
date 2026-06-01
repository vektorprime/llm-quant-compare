# GGUF Quantization Comparison

This directory contains a self-contained comparer for measuring value drift
between a reference GGUF and a quantized GGUF.

For Q8_0 tensors, the tool dequantizes every 32-value block as:

```text
final_weight = float16_scale * int8_weight
```

It then compares those final weights against the reference tensor values and
writes tensor, layer, and sublayer summaries.

## Run

Ubuntu setup:

```bash
python3 --version
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt
python compare_gguf_quant.py --help
```

The comparer needs Python 3.10 or newer. Runtime comparison needs `numpy`;
`tqdm` provides progress bars; `huggingface_hub` is included for downloading
GGUFs from Hugging Face.

Windows example:

```powershell
python .\compare_gguf_quant.py `
  --reference .\Qwen3.5-2B-BF16.gguf `
  --quant .\Qwen3.5-2B-Q8_0.gguf `
  --out-dir .\quant_compare_report
```

Each run writes into a fresh subfolder named after the model plus the local
date and time, for example
`quant_compare_report\Qwen3.5-2B-20260601-113012`. Existing reports are never
overwritten.

Split GGUFs are supported. Pass every shard, or use a glob:

Ubuntu example:

```bash
python compare_gguf_quant.py \
  --reference "./models/Qwen3.6-27B-MTP-GGUF/BF16/Qwen3.6-27B-BF16-"*.gguf \
  --quant "./models/Qwen3.6-27B-MTP-GGUF/Qwen3.6-27B-Q8_0.gguf" \
  --out-dir ./quant_compare_qwen36_27b_q8
```

Windows example:

```powershell
python .\compare_gguf_quant.py `
  --reference ".\models\Qwen3.6-27B-MTP-GGUF\BF16\Qwen3.6-27B-BF16-*.gguf" `
  --quant ".\models\Qwen3.6-27B-MTP-GGUF\Qwen3.6-27B-Q8_0.gguf" `
  --out-dir .\quant_compare_qwen36_27b_q8
```

To download the Qwen3.6 27B BF16 reference shards and Q8_0 candidate from
Hugging Face:

Ubuntu:

```bash
huggingface-cli download unsloth/Qwen3.6-27B-MTP-GGUF \
  BF16/Qwen3.6-27B-BF16-00001-of-00002.gguf \
  BF16/Qwen3.6-27B-BF16-00002-of-00002.gguf \
  Qwen3.6-27B-Q8_0.gguf \
  --local-dir ./models/Qwen3.6-27B-MTP-GGUF
```

Windows:

```powershell
huggingface-cli download unsloth/Qwen3.6-27B-MTP-GGUF `
  BF16/Qwen3.6-27B-BF16-00001-of-00002.gguf `
  BF16/Qwen3.6-27B-BF16-00002-of-00002.gguf `
  Qwen3.6-27B-Q8_0.gguf `
  --local-dir .\models\Qwen3.6-27B-MTP-GGUF
```

Each run folder contains:

- `report.md`: human-readable summary and worst affected tensors.
- `tensor_metrics.csv`: one row per tensor/sublayer.
- `layer_metrics.csv`: aggregate drift per `blk.N` layer.
- `sublayer_metrics.csv`: aggregate drift per sublayer name across blocks.
- `metrics.json`: machine-readable copy of all results.

## Key Metrics

- `relative_l2_error`: best single "badness" score. Higher means more quantization drift.
- `snr_db`: signal-to-noise ratio in dB. Lower means more quantization drift.
- `rmse` and `mae`: absolute error magnitudes in the tensor's weight units.
- `cosine_similarity`: directional agreement between reference and dequantized weights.
- `max_abs_error`: largest single-value difference in the tensor.

The script streams from memory-mapped files, so it does not load the full
multi-GB GGUFs into RAM.
