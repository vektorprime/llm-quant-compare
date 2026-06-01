# GGUF Quantization Comparison
- Reference: `Qwen3.5-2B-BF16.gguf`
- Candidate: `Qwen3.5-2B-Q8_0.gguf`
- Compared tensors: 320
- Compared elements: 1,881,825,088
- Elapsed: 37.74 seconds
- Reference tensor types: `{"BF16": 187, "F32": 133}`
- Candidate tensor types: `{"F32": 133, "Q8_0": 187}`

Q8_0 tensors are dequantized as `final_weight = float16_scale * int8_weight` for every 32-value block before comparison.

## Overall

- Relative L2 error: `0.00446639`
- SNR dB: `47.0009`
- RMSE: `7.210344e-05`
- MAE: `5.700279e-05`
- Max absolute error: `0.0034256`

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

| tensor | type | elements | rel_l2 | snr_db | rmse | mae | max_abs |
| --- | --- | --- | --- | --- | --- | --- | --- |
| blk.19.attn_v.weight | Q8_0 | 1048576 | 0.00772622 | 42.2407 | 1.663717e-04 | 1.252567e-04 | 0.00211716 |
| blk.23.attn_k.weight | Q8_0 | 1048576 | 0.00730368 | 42.7292 | 7.457965e-05 | 5.587079e-05 | 7.209778e-04 |
| blk.4.ssm_beta.weight | Q8_0 | 32768 | 0.00711364 | 42.9582 | 7.363473e-05 | 5.629565e-05 | 4.882812e-04 |
| blk.17.ssm_beta.weight | Q8_0 | 32768 | 0.00706985 | 43.0118 | 8.098961e-05 | 6.221661e-05 | 4.572868e-04 |
| blk.23.attn_v.weight | Q8_0 | 1048576 | 0.00705984 | 43.0241 | 1.421630e-04 | 1.059276e-04 | 0.00138855 |
| blk.22.ssm_beta.weight | Q8_0 | 32768 | 0.00698661 | 43.1147 | 9.625436e-05 | 7.388957e-05 | 7.410049e-04 |
| blk.6.ssm_beta.weight | Q8_0 | 32768 | 0.00698614 | 43.1153 | 8.234130e-05 | 6.385887e-05 | 5.483627e-04 |
| blk.5.ssm_beta.weight | Q8_0 | 32768 | 0.00698584 | 43.1156 | 7.422690e-05 | 5.707834e-05 | 3.972054e-04 |
| blk.21.ssm_beta.weight | Q8_0 | 32768 | 0.00697381 | 43.1306 | 8.430994e-05 | 6.454467e-05 | 4.506111e-04 |
| blk.2.ssm_beta.weight | Q8_0 | 32768 | 0.00693478 | 43.1794 | 8.923082e-05 | 6.741932e-05 | 5.474091e-04 |
| blk.20.ssm_beta.weight | Q8_0 | 32768 | 0.00691243 | 43.2074 | 9.313357e-05 | 7.083210e-05 | 6.380081e-04 |
| blk.15.attn_v.weight | Q8_0 | 1048576 | 0.00690499 | 43.2167 | 1.175811e-04 | 8.992170e-05 | 0.0014286 |
| blk.18.ssm_beta.weight | Q8_0 | 32768 | 0.00689492 | 43.2294 | 8.032944e-05 | 6.196544e-05 | 5.664825e-04 |
| blk.21.ssm_alpha.weight | Q8_0 | 32768 | 0.00687094 | 43.2597 | 1.472110e-04 | 1.134820e-04 | 8.420944e-04 |
| blk.8.ssm_beta.weight | Q8_0 | 32768 | 0.00680444 | 43.3442 | 6.405034e-05 | 4.990500e-05 | 3.261566e-04 |
| blk.22.ssm_alpha.weight | Q8_0 | 32768 | 0.00677413 | 43.3829 | 1.579280e-04 | 1.219618e-04 | 9.460449e-04 |
| blk.20.ssm_alpha.weight | Q8_0 | 32768 | 0.00674395 | 43.4217 | 1.386249e-04 | 1.075658e-04 | 8.049011e-04 |
| blk.16.ssm_beta.weight | Q8_0 | 32768 | 0.00673183 | 43.4373 | 8.074629e-05 | 6.072642e-05 | 7.963181e-04 |
| blk.17.ssm_alpha.weight | Q8_0 | 32768 | 0.00664516 | 43.5499 | 1.291345e-04 | 1.002601e-04 | 8.583069e-04 |
| blk.13.ssm_beta.weight | Q8_0 | 32768 | 0.00662942 | 43.5705 | 6.597718e-05 | 5.130843e-05 | 3.876686e-04 |
| blk.14.ssm_beta.weight | Q8_0 | 32768 | 0.00651724 | 43.7187 | 6.743939e-05 | 5.195965e-05 | 4.129410e-04 |
| blk.18.ssm_alpha.weight | Q8_0 | 32768 | 0.00651658 | 43.7196 | 1.184545e-04 | 9.179330e-05 | 7.209778e-04 |
| blk.1.ssm_beta.weight | Q8_0 | 32768 | 0.00649636 | 43.7466 | 9.388064e-05 | 7.166761e-05 | 5.855560e-04 |
| blk.19.attn_k.weight | Q8_0 | 1048576 | 0.00646475 | 43.789 | 7.209443e-05 | 5.624625e-05 | 7.095337e-04 |
| blk.23.attn_output.weight | Q8_0 | 4194304 | 0.00640426 | 43.8706 | 9.572022e-05 | 7.623898e-05 | 0.00141907 |

## Lowest SNR Tensors

| tensor | type | elements | rel_l2 | snr_db | rmse | mae |
| --- | --- | --- | --- | --- | --- | --- |
| blk.19.attn_v.weight | Q8_0 | 1048576 | 0.00772622 | 42.2407 | 1.663717e-04 | 1.252567e-04 |
| blk.23.attn_k.weight | Q8_0 | 1048576 | 0.00730368 | 42.7292 | 7.457965e-05 | 5.587079e-05 |
| blk.4.ssm_beta.weight | Q8_0 | 32768 | 0.00711364 | 42.9582 | 7.363473e-05 | 5.629565e-05 |
| blk.17.ssm_beta.weight | Q8_0 | 32768 | 0.00706985 | 43.0118 | 8.098961e-05 | 6.221661e-05 |
| blk.23.attn_v.weight | Q8_0 | 1048576 | 0.00705984 | 43.0241 | 1.421630e-04 | 1.059276e-04 |
| blk.22.ssm_beta.weight | Q8_0 | 32768 | 0.00698661 | 43.1147 | 9.625436e-05 | 7.388957e-05 |
| blk.6.ssm_beta.weight | Q8_0 | 32768 | 0.00698614 | 43.1153 | 8.234130e-05 | 6.385887e-05 |
| blk.5.ssm_beta.weight | Q8_0 | 32768 | 0.00698584 | 43.1156 | 7.422690e-05 | 5.707834e-05 |
| blk.21.ssm_beta.weight | Q8_0 | 32768 | 0.00697381 | 43.1306 | 8.430994e-05 | 6.454467e-05 |
| blk.2.ssm_beta.weight | Q8_0 | 32768 | 0.00693478 | 43.1794 | 8.923082e-05 | 6.741932e-05 |
| blk.20.ssm_beta.weight | Q8_0 | 32768 | 0.00691243 | 43.2074 | 9.313357e-05 | 7.083210e-05 |
| blk.15.attn_v.weight | Q8_0 | 1048576 | 0.00690499 | 43.2167 | 1.175811e-04 | 8.992170e-05 |
| blk.18.ssm_beta.weight | Q8_0 | 32768 | 0.00689492 | 43.2294 | 8.032944e-05 | 6.196544e-05 |
| blk.21.ssm_alpha.weight | Q8_0 | 32768 | 0.00687094 | 43.2597 | 1.472110e-04 | 1.134820e-04 |
| blk.8.ssm_beta.weight | Q8_0 | 32768 | 0.00680444 | 43.3442 | 6.405034e-05 | 4.990500e-05 |
| blk.22.ssm_alpha.weight | Q8_0 | 32768 | 0.00677413 | 43.3829 | 1.579280e-04 | 1.219618e-04 |
| blk.20.ssm_alpha.weight | Q8_0 | 32768 | 0.00674395 | 43.4217 | 1.386249e-04 | 1.075658e-04 |
| blk.16.ssm_beta.weight | Q8_0 | 32768 | 0.00673183 | 43.4373 | 8.074629e-05 | 6.072642e-05 |
| blk.17.ssm_alpha.weight | Q8_0 | 32768 | 0.00664516 | 43.5499 | 1.291345e-04 | 1.002601e-04 |
| blk.13.ssm_beta.weight | Q8_0 | 32768 | 0.00662942 | 43.5705 | 6.597718e-05 | 5.130843e-05 |
| blk.14.ssm_beta.weight | Q8_0 | 32768 | 0.00651724 | 43.7187 | 6.743939e-05 | 5.195965e-05 |
| blk.18.ssm_alpha.weight | Q8_0 | 32768 | 0.00651658 | 43.7196 | 1.184545e-04 | 9.179330e-05 |
| blk.1.ssm_beta.weight | Q8_0 | 32768 | 0.00649636 | 43.7466 | 9.388064e-05 | 7.166761e-05 |
| blk.19.attn_k.weight | Q8_0 | 1048576 | 0.00646475 | 43.789 | 7.209443e-05 | 5.624625e-05 |
| blk.23.attn_output.weight | Q8_0 | 4194304 | 0.00640426 | 43.8706 | 9.572022e-05 | 7.623898e-05 |

## Worst Layers

| layer | elements | rel_l2 | snr_db | rmse | mae |
| --- | --- | --- | --- | --- | --- |
| global | 508561408 | 0.00516072 | 45.7458 | 8.580232e-05 | 7.069746e-05 |
| blk.23 | 52433408 | 0.00436681 | 47.1967 | 7.635070e-05 | 5.974849e-05 |
| blk.14 | 58814624 | 0.00428881 | 47.3533 | 6.596501e-05 | 5.142663e-05 |
| blk.10 | 58814624 | 0.00423376 | 47.4655 | 6.136184e-05 | 4.836833e-05 |
| blk.16 | 58814624 | 0.00422726 | 47.4788 | 7.168922e-05 | 5.562321e-05 |
| blk.0 | 58814624 | 0.00422138 | 47.4909 | 6.167406e-05 | 4.852204e-05 |
| blk.6 | 58814624 | 0.00420873 | 47.517 | 6.291531e-05 | 4.989146e-05 |
| blk.13 | 58814624 | 0.00419498 | 47.5454 | 6.305767e-05 | 4.958956e-05 |
| blk.20 | 58814624 | 0.00419338 | 47.5487 | 7.547603e-05 | 5.845046e-05 |
| blk.22 | 58814624 | 0.0041808 | 47.5748 | 7.397691e-05 | 5.871266e-05 |
| blk.4 | 58814624 | 0.00417368 | 47.5896 | 6.477515e-05 | 5.082777e-05 |
| blk.12 | 58814624 | 0.004172 | 47.5931 | 6.420285e-05 | 4.991689e-05 |
| blk.9 | 58814624 | 0.00417131 | 47.5945 | 6.084520e-05 | 4.813270e-05 |
| blk.8 | 58814624 | 0.00416937 | 47.5986 | 6.185153e-05 | 4.880852e-05 |
| blk.5 | 58814624 | 0.00416603 | 47.6055 | 6.372757e-05 | 5.036890e-05 |
| blk.2 | 58814624 | 0.00416214 | 47.6137 | 6.428932e-05 | 5.089551e-05 |
| blk.1 | 58814624 | 0.00414831 | 47.6426 | 6.114495e-05 | 4.874363e-05 |
| blk.18 | 58814624 | 0.00413969 | 47.6606 | 7.558881e-05 | 5.848645e-05 |
| blk.17 | 58814624 | 0.0041365 | 47.6673 | 7.294809e-05 | 5.685182e-05 |
| blk.19 | 52433408 | 0.00411827 | 47.7057 | 7.031989e-05 | 5.501920e-05 |
| blk.21 | 58814624 | 0.00410305 | 47.7379 | 7.495860e-05 | 5.904267e-05 |
| blk.15 | 52433408 | 0.00402751 | 47.8993 | 6.349673e-05 | 5.029094e-05 |
| blk.3 | 52433408 | 0.003954 | 48.0593 | 6.064003e-05 | 4.771926e-05 |
| blk.7 | 52433408 | 0.00390446 | 48.1688 | 5.639589e-05 | 4.526092e-05 |
| blk.11 | 52433408 | 0.00386342 | 48.2606 | 5.579830e-05 | 4.469053e-05 |

## Worst Sublayers Across Blocks

| sublayer | elements | rel_l2 | snr_db | rmse | mae |
| --- | --- | --- | --- | --- | --- |
| attn_v.weight | 6291456 | 0.00689837 | 43.2251 | 1.147223e-04 | 8.409912e-05 |
| ssm_beta.weight | 589824 | 0.00673418 | 43.4343 | 7.994290e-05 | 6.122125e-05 |
| ssm_alpha.weight | 589824 | 0.00614405 | 44.2309 | 1.230473e-04 | 9.533450e-05 |
| attn_k.weight | 6291456 | 0.00613142 | 44.2488 | 7.005403e-05 | 5.508968e-05 |
| attn_output.weight | 25165824 | 0.00578953 | 44.7471 | 8.084282e-05 | 6.418438e-05 |
| token_embd.weight | 508559360 | 0.00571714 | 44.8564 | 8.580250e-05 | 7.069774e-05 |
| attn_q.weight | 50331648 | 0.00566762 | 44.932 | 7.671955e-05 | 6.125973e-05 |
| ssm_out.weight | 75497472 | 0.00566722 | 44.9326 | 6.643220e-05 | 5.362415e-05 |
| attn_qkv.weight | 226492416 | 0.00564444 | 44.9676 | 9.373098e-05 | 7.503815e-05 |
| ffn_gate.weight | 301989888 | 0.00556192 | 45.0955 | 6.425652e-05 | 5.152985e-05 |
| attn_gate.weight | 75497472 | 0.00555879 | 45.1004 | 7.474946e-05 | 6.033534e-05 |
| ffn_down.weight | 301989888 | 0.00555707 | 45.1031 | 4.819682e-05 | 3.958375e-05 |
| ffn_up.weight | 301989888 | 0.00551772 | 45.1648 | 5.013372e-05 | 4.150493e-05 |
| attn_k_norm.weight | 1536 | 0 | inf | 0 | 0 |
| attn_norm.weight | 49152 | 0 | inf | 0 | 0 |
| attn_q_norm.weight | 1536 | 0 | inf | 0 | 0 |
| output_norm.weight | 2048 | 0 | inf | 0 | 0 |
| post_attention_norm.weight | 49152 | 0 | inf | 0 | 0 |
| ssm_a | 288 | 0 | inf | 0 | 0 |
| ssm_conv1d.weight | 442368 | 0 | inf | 0 | 0 |
| ssm_dt.bias | 288 | 0 | inf | 0 | 0 |
| ssm_norm.weight | 2304 | 0 | inf | 0 | 0 |
