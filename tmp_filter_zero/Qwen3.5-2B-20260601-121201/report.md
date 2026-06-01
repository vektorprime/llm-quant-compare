# GGUF Quantization Comparison
- Output directory: `tmp_filter_zero\Qwen3.5-2B-20260601-121201`
- Reference: `Qwen3.5-2B-BF16.gguf`
- Candidate: `Qwen3.5-2B-Q8_0.gguf`
- Reference files: 1
- Candidate files: 1
- Compared tensors: 1
- Compared elements: 2,048
- Elapsed: 0.14 seconds
- Reference tensor types: `{"BF16": 187, "F32": 133}`
- Candidate tensor types: `{"F32": 133, "Q8_0": 187}`
- Zero-error tensor rows hidden from ranked tables: 1

Q8_0 tensors are dequantized as `final_weight = float16_scale * int8_weight` for every 32-value block before comparison.

## Overall

- Relative L2 error: `0`
- SNR dB: `inf`
- RMSE: `0`
- MAE: `0`
- Max absolute error: `0`

Higher relative L2 error and lower SNR identify tensors that were more negatively affected by quantization.

## How To Read The Columns

Notation used below:

- `ref`: the reference/native value from the BF16/F32 GGUF.
- `quant`: the candidate value after dequantization. For Q8_0 this is `float16_scale * int8_weight` for each value in a 32-value block.
- `error`: `quant - ref`.
- `n`: number of values included in that tensor, layer, or sublayer group.

Layer and sublayer rows are computed by aggregating the underlying sums across all matching tensors, then computing the metric from those totals. They are not simple averages of the per-tensor rows.

| Column | Meaning | Computation |
| --- | --- | --- |
| `tensor` / `name` | Tensor name from the GGUF metadata. | Matched by exact tensor name between the two files. |
| `layer` | Transformer block group, such as `blk.23`, or `global` for tensors outside `blk.N.*`. | Parsed from the tensor name. |
| `layer_index` | Numeric block index when the tensor is inside `blk.N.*`. | Parsed from the tensor name; blank for `global`. |
| `sublayer` | The part of the tensor name after `blk.N.`, such as `attn_v.weight`. | Parsed from the tensor name; global tensors use the full tensor name. |
| `shape` | Tensor dimensions as stored in GGUF. | Read from tensor metadata. |
| `type` / `quant_type` | Candidate GGML tensor type. | Read from the quantized GGUF metadata. |
| `ref_type` | Reference GGML tensor type. | Read from the reference GGUF metadata. |
| `ref_file` | Reference shard containing this tensor. Useful for split GGUFs. | Source path where the matched reference tensor was found. |
| `quant_file` | Candidate shard containing this tensor. Useful for split GGUFs. | Source path where the matched candidate tensor was found. |
| `elements` | Number of scalar values compared. | Product of tensor dimensions, or sum of elements for grouped rows. |
| `ref_bytes` | On-disk bytes used by the reference tensor(s). | Computed from tensor type and element count. |
| `quant_bytes` | On-disk bytes used by the candidate tensor(s). | Computed from tensor type and element count. Q8_0 uses 34 bytes per 32 values. |
| `compression_ratio` | Storage reduction from reference to candidate. Higher is smaller candidate storage. | `ref_bytes / quant_bytes`. |
| `mean_ref` | Average reference value. | `sum(ref) / n`. |
| `mean_quant` | Average candidate value after dequantization. | `sum(quant) / n`. |
| `mean_error` | Average signed drift. Positive means the candidate is larger on average. | `sum(error) / n`. |
| `mean_abs_ref` | Average absolute reference magnitude. | `sum(abs(ref)) / n`. |
| `mean_abs_quant` | Average absolute candidate magnitude. | `sum(abs(quant)) / n`. |
| `mae` | Mean absolute error. This is the average absolute value drift in weight units. Lower is better. | `sum(abs(error)) / n`. |
| `mean_abs_relative_error` | MAE normalized by the average reference magnitude. Lower is better. | `mae / mean_abs_ref`. |
| `rmse` | Root mean squared error. This penalizes large individual errors more than MAE. Lower is better. | `sqrt(sum(error^2) / n)`. |
| `rms_ref` | Root mean square magnitude of the reference values. | `sqrt(sum(ref^2) / n)`. |
| `rel_l2` / `relative_l2_error` | Error-vector size relative to the reference-vector size. This is usually the best single "badness" score. Lower is better; `0.01` means the error vector is about 1% as large as the reference vector. | `sqrt(sum(error^2) / sum(ref^2))`. |
| `cosine_similarity` | Directional agreement between reference and candidate vectors. Closer to `1` is better. | `sum(ref * quant) / sqrt(sum(ref^2) * sum(quant^2))`. |
| `snr_db` | Signal-to-noise ratio in decibels. Higher is better; `inf` means no measured error. | `10 * log10(sum(ref^2) / sum(error^2))`. |
| `max_abs_error` / `max_abs` | Largest single absolute difference. Lower is better. | `max(abs(error))`. |
| `max_abs_error_index` | Flattened element index where `max_abs_error` occurred. | Index from row-major flattening of the tensor values. |
| `ref_at_max_abs_error` | Reference value at the largest-error element. | `ref[max_abs_error_index]`. |
| `quant_at_max_abs_error` | Candidate value at the largest-error element. | `quant[max_abs_error_index]`. |

Quick interpretation: sort by highest `rel_l2` or lowest `snr_db` to find tensors most affected by quantization. Use `rmse`, `mae`, and `max_abs_error` to understand the absolute size of the drift.

## Worst Tensors By Relative L2 Error

_None._

## Lowest SNR Tensors

_None._

## Worst Layers

_None._

## Worst Sublayers Across Blocks

_None._
