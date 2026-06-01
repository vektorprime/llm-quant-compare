#!/usr/bin/env python3
"""Compare value drift between a reference GGUF and a quantized GGUF.

This tool is intentionally self-contained: it parses the GGUF header directly,
memory maps tensor bytes, dequantizes Q8_0 blocks as ``scale * int8`` values,
and streams tensor chunks through NumPy so multi-GB models do not need to be
loaded into RAM at once.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import mmap
import os
import re
import struct
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO, Iterable

import numpy as np

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - tqdm is optional at runtime.
    tqdm = None


GGUF_MAGIC = b"GGUF"

GGUF_VALUE_TYPE_NAMES = {
    0: "UINT8",
    1: "INT8",
    2: "UINT16",
    3: "INT16",
    4: "UINT32",
    5: "INT32",
    6: "FLOAT32",
    7: "BOOL",
    8: "STRING",
    9: "ARRAY",
    10: "UINT64",
    11: "INT64",
    12: "FLOAT64",
}

GGUF_SCALAR_FORMATS = {
    0: "<B",
    1: "<b",
    2: "<H",
    3: "<h",
    4: "<I",
    5: "<i",
    6: "<f",
    7: "<?",
    10: "<Q",
    11: "<q",
    12: "<d",
}

GGML_TYPE_NAMES = {
    0: "F32",
    1: "F16",
    2: "Q4_0",
    3: "Q4_1",
    6: "Q5_0",
    7: "Q5_1",
    8: "Q8_0",
    9: "Q8_1",
    10: "Q2_K",
    11: "Q3_K",
    12: "Q4_K",
    13: "Q5_K",
    14: "Q6_K",
    15: "Q8_K",
    16: "IQ2_XXS",
    17: "IQ2_XS",
    18: "IQ3_XXS",
    19: "IQ1_S",
    20: "IQ4_NL",
    21: "IQ3_S",
    22: "IQ2_S",
    23: "IQ4_XS",
    24: "I8",
    25: "I16",
    26: "I32",
    27: "I64",
    28: "F64",
    29: "IQ1_M",
    30: "BF16",
    31: "Q4_0_4_4",
    32: "Q4_0_4_8",
    33: "Q4_0_8_8",
    34: "TQ1_0",
    35: "TQ2_0",
    36: "MXFP4",
}

GGML_F32 = 0
GGML_F16 = 1
GGML_Q8_0 = 8
GGML_I8 = 24
GGML_I16 = 25
GGML_I32 = 26
GGML_I64 = 27
GGML_F64 = 28
GGML_BF16 = 30

DENSE_NUMPY_TYPES = {
    GGML_F32: np.dtype("<f4"),
    GGML_F16: np.dtype("<f2"),
    GGML_BF16: np.dtype("<u2"),
    GGML_I8: np.dtype("i1"),
    GGML_I16: np.dtype("<i2"),
    GGML_I32: np.dtype("<i4"),
    GGML_I64: np.dtype("<i8"),
    GGML_F64: np.dtype("<f8"),
}

Q8_0_BLOCK_SIZE = 32
Q8_0_DTYPE = np.dtype([("d", "<f2"), ("qs", "i1", (Q8_0_BLOCK_SIZE,))])

SUPPORTED_TYPES = set(DENSE_NUMPY_TYPES) | {GGML_Q8_0}

SELECTED_METADATA_KEYS = {
    "general.architecture",
    "general.name",
    "general.file_type",
    "general.quantization_version",
    "general.alignment",
}

BLOCK_NAME_RE = re.compile(r"^blk\.(\d+)\.(.+)$")


def align_to(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def read_exact(f: BinaryIO, n: int) -> bytes:
    data = f.read(n)
    if len(data) != n:
        raise EOFError(f"expected {n} bytes, got {len(data)}")
    return data


def read_u32(f: BinaryIO) -> int:
    return struct.unpack("<I", read_exact(f, 4))[0]


def read_u64(f: BinaryIO) -> int:
    return struct.unpack("<Q", read_exact(f, 8))[0]


def read_gguf_string(f: BinaryIO) -> str:
    n = read_u64(f)
    return read_exact(f, n).decode("utf-8", errors="replace")


def skip_gguf_value(f: BinaryIO, value_type: int) -> None:
    if value_type == 8:
        f.seek(read_u64(f), os.SEEK_CUR)
        return

    if value_type == 9:
        array_type = read_u32(f)
        array_len = read_u64(f)
        if array_type == 8:
            for _ in range(array_len):
                f.seek(read_u64(f), os.SEEK_CUR)
            return
        if array_type in GGUF_SCALAR_FORMATS:
            f.seek(struct.calcsize(GGUF_SCALAR_FORMATS[array_type]) * array_len, os.SEEK_CUR)
            return
        raise ValueError(
            f"unsupported GGUF metadata array type {array_type} "
            f"({GGUF_VALUE_TYPE_NAMES.get(array_type, 'unknown')})"
        )

    if value_type in GGUF_SCALAR_FORMATS:
        f.seek(struct.calcsize(GGUF_SCALAR_FORMATS[value_type]), os.SEEK_CUR)
        return

    raise ValueError(
        f"unsupported GGUF metadata value type {value_type} "
        f"({GGUF_VALUE_TYPE_NAMES.get(value_type, 'unknown')})"
    )


def read_selected_metadata_value(f: BinaryIO, value_type: int) -> Any:
    if value_type == 8:
        return read_gguf_string(f)
    if value_type in GGUF_SCALAR_FORMATS:
        fmt = GGUF_SCALAR_FORMATS[value_type]
        return struct.unpack(fmt, read_exact(f, struct.calcsize(fmt)))[0]
    skip_gguf_value(f, value_type)
    return None


def ggml_type_name(type_id: int) -> str:
    return GGML_TYPE_NAMES.get(type_id, f"TYPE_{type_id}")


@dataclass
class TensorInfo:
    name: str
    shape: tuple[int, ...]
    ggml_type: int
    relative_offset: int
    data_offset: int = 0

    @property
    def n_elements(self) -> int:
        return math.prod(self.shape)

    @property
    def type_name(self) -> str:
        return ggml_type_name(self.ggml_type)

    @property
    def shape_text(self) -> str:
        return "x".join(str(x) for x in self.shape)

    @property
    def n_bytes(self) -> int:
        return tensor_nbytes(self)


@dataclass
class GGUFFile:
    path: Path
    version: int
    tensor_count: int
    metadata_count: int
    alignment: int
    metadata: dict[str, Any]
    tensors: list[TensorInfo]
    data_start: int

    @property
    def tensors_by_name(self) -> dict[str, TensorInfo]:
        return {tensor.name: tensor for tensor in self.tensors}


def tensor_nbytes(tensor: TensorInfo) -> int:
    n = tensor.n_elements
    if tensor.ggml_type == GGML_Q8_0:
        if n % Q8_0_BLOCK_SIZE:
            raise ValueError(f"{tensor.name}: Q8_0 element count {n} is not divisible by 32")
        return (n // Q8_0_BLOCK_SIZE) * Q8_0_DTYPE.itemsize
    if tensor.ggml_type in DENSE_NUMPY_TYPES:
        return n * DENSE_NUMPY_TYPES[tensor.ggml_type].itemsize
    raise ValueError(f"{tensor.name}: unsupported tensor type {tensor.type_name}")


def parse_gguf(path: Path) -> GGUFFile:
    with path.open("rb") as f:
        magic = read_exact(f, 4)
        if magic != GGUF_MAGIC:
            raise ValueError(f"{path} is not a GGUF file, magic={magic!r}")

        version = read_u32(f)
        tensor_count = read_u64(f)
        metadata_count = read_u64(f)
        metadata: dict[str, Any] = {}
        alignment = 32

        for _ in range(metadata_count):
            key = read_gguf_string(f)
            value_type = read_u32(f)
            if key in SELECTED_METADATA_KEYS:
                value = read_selected_metadata_value(f, value_type)
                metadata[key] = value
                if key == "general.alignment" and value:
                    alignment = int(value)
            else:
                skip_gguf_value(f, value_type)

        tensors: list[TensorInfo] = []
        for _ in range(tensor_count):
            name = read_gguf_string(f)
            n_dimensions = read_u32(f)
            shape = tuple(read_u64(f) for _ in range(n_dimensions))
            ggml_type = read_u32(f)
            relative_offset = read_u64(f)
            tensors.append(TensorInfo(name, shape, ggml_type, relative_offset))

        data_start = align_to(f.tell(), alignment)
        file_size = path.stat().st_size
        for tensor in tensors:
            tensor.data_offset = data_start + tensor.relative_offset
            if tensor.ggml_type in SUPPORTED_TYPES:
                end = tensor.data_offset + tensor.n_bytes
                if end > file_size:
                    raise ValueError(
                        f"{path}: tensor {tensor.name} extends past EOF "
                        f"({end} > {file_size})"
                    )

        return GGUFFile(
            path=path,
            version=version,
            tensor_count=tensor_count,
            metadata_count=metadata_count,
            alignment=alignment,
            metadata=metadata,
            tensors=tensors,
            data_start=data_start,
        )


def open_mmap(path: Path) -> tuple[Any, mmap.mmap]:
    f = path.open("rb")
    try:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    except Exception:
        f.close()
        raise
    return f, mm


def dequant_q8_0(mm: mmap.mmap, tensor: TensorInfo, block_start: int, block_count: int) -> np.ndarray:
    byte_offset = tensor.data_offset + block_start * Q8_0_DTYPE.itemsize
    blocks = np.frombuffer(mm, dtype=Q8_0_DTYPE, count=block_count, offset=byte_offset)
    scales = blocks["d"].astype(np.float32).reshape(-1, 1)
    qs = blocks["qs"].astype(np.float32)
    return np.multiply(qs, scales, dtype=np.float32).reshape(-1)


def read_dense_values(mm: mmap.mmap, tensor: TensorInfo, start: int, count: int) -> np.ndarray:
    dtype = DENSE_NUMPY_TYPES[tensor.ggml_type]
    byte_offset = tensor.data_offset + start * dtype.itemsize
    raw = np.frombuffer(mm, dtype=dtype, count=count, offset=byte_offset)

    if tensor.ggml_type == GGML_BF16:
        bits = raw.astype(np.uint32)
        bits <<= 16
        return bits.view(np.float32)

    if tensor.ggml_type in (GGML_F16, GGML_I8, GGML_I16, GGML_I32, GGML_I64, GGML_F64):
        return raw.astype(np.float32)

    return raw.astype(np.float32, copy=False)


def read_values(mm: mmap.mmap, tensor: TensorInfo, start: int, count: int) -> np.ndarray:
    if tensor.ggml_type == GGML_Q8_0:
        if start % Q8_0_BLOCK_SIZE or count % Q8_0_BLOCK_SIZE:
            raise ValueError(f"{tensor.name}: Q8_0 reads must be 32-value aligned")
        return dequant_q8_0(mm, tensor, start // Q8_0_BLOCK_SIZE, count // Q8_0_BLOCK_SIZE)
    return read_dense_values(mm, tensor, start, count)


@dataclass
class RunningStats:
    n: int = 0
    sum_ref: float = 0.0
    sum_quant: float = 0.0
    sum_abs_ref: float = 0.0
    sum_abs_quant: float = 0.0
    sum_error: float = 0.0
    sum_abs_error: float = 0.0
    sum_squared_error: float = 0.0
    sum_squared_ref: float = 0.0
    sum_squared_quant: float = 0.0
    dot: float = 0.0
    max_abs_error: float = -1.0
    max_abs_error_index: int = -1
    ref_at_max_abs_error: float = 0.0
    quant_at_max_abs_error: float = 0.0

    def update(self, ref: np.ndarray, quant: np.ndarray, start_index: int) -> None:
        ref = ref.reshape(-1)
        quant = quant.reshape(-1)
        if ref.shape != quant.shape:
            raise ValueError(f"shape mismatch in chunk: {ref.shape} vs {quant.shape}")

        error = np.subtract(quant, ref, dtype=np.float32)
        abs_ref = np.abs(ref)
        abs_quant = np.abs(quant)
        abs_error = np.abs(error)

        self.n += int(ref.size)
        self.sum_ref += float(np.sum(ref, dtype=np.float64))
        self.sum_quant += float(np.sum(quant, dtype=np.float64))
        self.sum_abs_ref += float(np.sum(abs_ref, dtype=np.float64))
        self.sum_abs_quant += float(np.sum(abs_quant, dtype=np.float64))
        self.sum_error += float(np.sum(error, dtype=np.float64))
        self.sum_abs_error += float(np.sum(abs_error, dtype=np.float64))
        self.sum_squared_error += float(
            np.sum(np.multiply(error, error, dtype=np.float64), dtype=np.float64)
        )
        self.sum_squared_ref += float(
            np.sum(np.multiply(ref, ref, dtype=np.float64), dtype=np.float64)
        )
        self.sum_squared_quant += float(
            np.sum(np.multiply(quant, quant, dtype=np.float64), dtype=np.float64)
        )
        self.dot += float(np.sum(np.multiply(ref, quant, dtype=np.float64), dtype=np.float64))

        local_index = int(np.argmax(abs_error))
        local_max = float(abs_error[local_index])
        if local_max > self.max_abs_error:
            self.max_abs_error = local_max
            self.max_abs_error_index = start_index + local_index
            self.ref_at_max_abs_error = float(ref[local_index])
            self.quant_at_max_abs_error = float(quant[local_index])

    def merge(self, other: "RunningStats") -> None:
        self.n += other.n
        self.sum_ref += other.sum_ref
        self.sum_quant += other.sum_quant
        self.sum_abs_ref += other.sum_abs_ref
        self.sum_abs_quant += other.sum_abs_quant
        self.sum_error += other.sum_error
        self.sum_abs_error += other.sum_abs_error
        self.sum_squared_error += other.sum_squared_error
        self.sum_squared_ref += other.sum_squared_ref
        self.sum_squared_quant += other.sum_squared_quant
        self.dot += other.dot
        if other.max_abs_error > self.max_abs_error:
            self.max_abs_error = other.max_abs_error
            self.max_abs_error_index = other.max_abs_error_index
            self.ref_at_max_abs_error = other.ref_at_max_abs_error
            self.quant_at_max_abs_error = other.quant_at_max_abs_error

    def metrics(self) -> dict[str, float | int]:
        if self.n == 0:
            return {
                "elements": 0,
                "mean_ref": math.nan,
                "mean_quant": math.nan,
                "mean_error": math.nan,
                "mean_abs_ref": math.nan,
                "mean_abs_quant": math.nan,
                "mae": math.nan,
                "mean_abs_relative_error": math.nan,
                "rmse": math.nan,
                "rms_ref": math.nan,
                "relative_l2_error": math.nan,
                "cosine_similarity": math.nan,
                "snr_db": math.nan,
                "max_abs_error": math.nan,
                "max_abs_error_index": -1,
                "ref_at_max_abs_error": math.nan,
                "quant_at_max_abs_error": math.nan,
            }

        eps = 1e-30
        mean_abs_ref = self.sum_abs_ref / self.n
        rmse = math.sqrt(self.sum_squared_error / self.n)
        rms_ref = math.sqrt(self.sum_squared_ref / self.n)
        relative_l2 = math.sqrt(self.sum_squared_error / max(self.sum_squared_ref, eps))
        denom = math.sqrt(max(self.sum_squared_ref, eps) * max(self.sum_squared_quant, eps))
        cosine = self.dot / denom
        if self.sum_squared_error == 0:
            snr_db = math.inf
        else:
            snr_db = 10.0 * math.log10(max(self.sum_squared_ref, eps) / self.sum_squared_error)

        return {
            "elements": self.n,
            "mean_ref": self.sum_ref / self.n,
            "mean_quant": self.sum_quant / self.n,
            "mean_error": self.sum_error / self.n,
            "mean_abs_ref": mean_abs_ref,
            "mean_abs_quant": self.sum_abs_quant / self.n,
            "mae": self.sum_abs_error / self.n,
            "mean_abs_relative_error": (self.sum_abs_error / self.n) / max(mean_abs_ref, eps),
            "rmse": rmse,
            "rms_ref": rms_ref,
            "relative_l2_error": relative_l2,
            "cosine_similarity": cosine,
            "snr_db": snr_db,
            "max_abs_error": self.max_abs_error,
            "max_abs_error_index": self.max_abs_error_index,
            "ref_at_max_abs_error": self.ref_at_max_abs_error,
            "quant_at_max_abs_error": self.quant_at_max_abs_error,
        }


def split_layer_sublayer(name: str) -> tuple[str, str, int | None]:
    match = BLOCK_NAME_RE.match(name)
    if match:
        layer_index = int(match.group(1))
        return f"blk.{layer_index}", match.group(2), layer_index
    return "global", name, None


def compare_tensor(
    ref_mm: mmap.mmap,
    quant_mm: mmap.mmap,
    ref_tensor: TensorInfo,
    quant_tensor: TensorInfo,
    chunk_blocks: int,
    chunk_values: int,
    progress: Any | None,
) -> RunningStats:
    stats = RunningStats()
    n = ref_tensor.n_elements

    if quant_tensor.ggml_type == GGML_Q8_0 or ref_tensor.ggml_type == GGML_Q8_0:
        if n % Q8_0_BLOCK_SIZE:
            raise ValueError(f"{ref_tensor.name}: element count {n} is not divisible by 32")
        total_blocks = n // Q8_0_BLOCK_SIZE
        for block_start in range(0, total_blocks, chunk_blocks):
            block_count = min(chunk_blocks, total_blocks - block_start)
            start = block_start * Q8_0_BLOCK_SIZE
            count = block_count * Q8_0_BLOCK_SIZE
            ref = read_values(ref_mm, ref_tensor, start, count)
            quant = read_values(quant_mm, quant_tensor, start, count)
            stats.update(ref, quant, start)
            if progress is not None:
                progress.update(count)
    else:
        for start in range(0, n, chunk_values):
            count = min(chunk_values, n - start)
            ref = read_values(ref_mm, ref_tensor, start, count)
            quant = read_values(quant_mm, quant_tensor, start, count)
            stats.update(ref, quant, start)
            if progress is not None:
                progress.update(count)

    return stats


def passes_filters(name: str, include: re.Pattern[str] | None, exclude: re.Pattern[str] | None) -> bool:
    if include is not None and include.search(name) is None:
        return False
    if exclude is not None and exclude.search(name) is not None:
        return False
    return True


def make_tensor_row(ref_tensor: TensorInfo, quant_tensor: TensorInfo, stats: RunningStats) -> dict[str, Any]:
    layer, sublayer, layer_index = split_layer_sublayer(ref_tensor.name)
    metrics = stats.metrics()
    row: dict[str, Any] = {
        "name": ref_tensor.name,
        "layer": layer,
        "layer_index": layer_index,
        "sublayer": sublayer,
        "shape": ref_tensor.shape_text,
        "ref_type": ref_tensor.type_name,
        "quant_type": quant_tensor.type_name,
        "ref_bytes": ref_tensor.n_bytes,
        "quant_bytes": quant_tensor.n_bytes,
        "compression_ratio": ref_tensor.n_bytes / quant_tensor.n_bytes
        if quant_tensor.n_bytes
        else math.nan,
    }
    row.update(metrics)
    return row


def aggregate_rows(rows: Iterable[dict[str, Any]], key_field: str) -> list[dict[str, Any]]:
    groups: dict[str, RunningStats] = {}
    bytes_by_group: dict[str, list[int]] = {}
    for row in rows:
        key = str(row[key_field])
        stats = RunningStats(
            n=int(row["elements"]),
            sum_ref=float(row["_sum_ref"]),
            sum_quant=float(row["_sum_quant"]),
            sum_abs_ref=float(row["_sum_abs_ref"]),
            sum_abs_quant=float(row["_sum_abs_quant"]),
            sum_error=float(row["_sum_error"]),
            sum_abs_error=float(row["_sum_abs_error"]),
            sum_squared_error=float(row["_sum_squared_error"]),
            sum_squared_ref=float(row["_sum_squared_ref"]),
            sum_squared_quant=float(row["_sum_squared_quant"]),
            dot=float(row["_dot"]),
            max_abs_error=float(row["max_abs_error"]),
            max_abs_error_index=int(row["max_abs_error_index"]),
            ref_at_max_abs_error=float(row["ref_at_max_abs_error"]),
            quant_at_max_abs_error=float(row["quant_at_max_abs_error"]),
        )
        groups.setdefault(key, RunningStats()).merge(stats)
        bytes_by_group.setdefault(key, [0, 0])
        bytes_by_group[key][0] += int(row["ref_bytes"])
        bytes_by_group[key][1] += int(row["quant_bytes"])

    out: list[dict[str, Any]] = []
    for key, stats in groups.items():
        ref_bytes, quant_bytes = bytes_by_group[key]
        metrics = stats.metrics()
        out.append(
            {
                key_field: key,
                "ref_bytes": ref_bytes,
                "quant_bytes": quant_bytes,
                "compression_ratio": ref_bytes / quant_bytes if quant_bytes else math.nan,
                **metrics,
            }
        )
    out.sort(key=lambda r: (-float(r["relative_l2_error"]), str(r[key_field])))
    return out


def stats_private_fields(stats: RunningStats) -> dict[str, float]:
    return {
        "_sum_ref": stats.sum_ref,
        "_sum_quant": stats.sum_quant,
        "_sum_abs_ref": stats.sum_abs_ref,
        "_sum_abs_quant": stats.sum_abs_quant,
        "_sum_error": stats.sum_error,
        "_sum_abs_error": stats.sum_abs_error,
        "_sum_squared_error": stats.sum_squared_error,
        "_sum_squared_ref": stats.sum_squared_ref,
        "_sum_squared_quant": stats.sum_squared_quant,
        "_dot": stats.dot,
    }


def public_row(row: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in row.items() if not k.startswith("_")}


def csv_value(value: Any) -> Any:
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
    return value


def write_csv(path: Path, rows: list[dict[str, Any]], field_order: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=field_order, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: csv_value(row.get(field, "")) for field in field_order})


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float):
        if math.isnan(value):
            return None
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
    return value


def fmt_float(value: Any, digits: int = 6) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        x = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(x):
        return "nan"
    if math.isinf(x):
        return "inf" if x > 0 else "-inf"
    if x == 0:
        return "0"
    if abs(x) >= 10000 or abs(x) < 0.001:
        return f"{x:.{digits}e}"
    return f"{x:.{digits}g}"


def markdown_table(rows: list[dict[str, Any]], columns: list[tuple[str, str]], limit: int) -> str:
    selected = rows[:limit]
    if not selected:
        return "_None._\n"
    header = "| " + " | ".join(label for label, _ in columns) + " |"
    sep = "| " + " | ".join("---" for _ in columns) + " |"
    lines = [header, sep]
    for row in selected:
        values = []
        for _, key in columns:
            value = row.get(key, "")
            if isinstance(value, float):
                value = fmt_float(value)
            values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def type_counts(gguf: GGUFFile) -> dict[str, int]:
    counts: dict[str, int] = {}
    for tensor in gguf.tensors:
        counts[tensor.type_name] = counts.get(tensor.type_name, 0) + 1
    return dict(sorted(counts.items()))


def metric_reference_markdown() -> str:
    return """## How To Read The Columns

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

"""


def write_markdown_report(
    path: Path,
    args: argparse.Namespace,
    ref: GGUFFile,
    quant: GGUFFile,
    tensor_rows: list[dict[str, Any]],
    layer_rows: list[dict[str, Any]],
    sublayer_rows: list[dict[str, Any]],
    skipped: list[dict[str, Any]],
    elapsed_s: float,
) -> None:
    total_stats = RunningStats()
    for row in tensor_rows:
        total_stats.merge(
            RunningStats(
                n=int(row["elements"]),
                sum_ref=float(row["_sum_ref"]),
                sum_quant=float(row["_sum_quant"]),
                sum_abs_ref=float(row["_sum_abs_ref"]),
                sum_abs_quant=float(row["_sum_abs_quant"]),
                sum_error=float(row["_sum_error"]),
                sum_abs_error=float(row["_sum_abs_error"]),
                sum_squared_error=float(row["_sum_squared_error"]),
                sum_squared_ref=float(row["_sum_squared_ref"]),
                sum_squared_quant=float(row["_sum_squared_quant"]),
                dot=float(row["_dot"]),
                max_abs_error=float(row["max_abs_error"]),
                max_abs_error_index=int(row["max_abs_error_index"]),
                ref_at_max_abs_error=float(row["ref_at_max_abs_error"]),
                quant_at_max_abs_error=float(row["quant_at_max_abs_error"]),
            )
        )
    total_metrics = total_stats.metrics()

    worst_tensors = sorted(tensor_rows, key=lambda r: -float(r["relative_l2_error"]))
    lowest_snr = sorted(tensor_rows, key=lambda r: float(r["snr_db"]))

    text = []
    text.append("# GGUF Quantization Comparison\n")
    text.append(f"- Reference: `{ref.path}`\n")
    text.append(f"- Candidate: `{quant.path}`\n")
    text.append(f"- Compared tensors: {len(tensor_rows)}\n")
    text.append(f"- Compared elements: {total_metrics['elements']:,}\n")
    text.append(f"- Elapsed: {elapsed_s:.2f} seconds\n")
    text.append(f"- Reference tensor types: `{json.dumps(type_counts(ref), sort_keys=True)}`\n")
    text.append(f"- Candidate tensor types: `{json.dumps(type_counts(quant), sort_keys=True)}`\n")
    text.append("\n")
    text.append("Q8_0 tensors are dequantized as `final_weight = float16_scale * int8_weight` for every 32-value block before comparison.\n")
    text.append("\n")
    text.append("## Overall\n\n")
    text.append(f"- Relative L2 error: `{fmt_float(total_metrics['relative_l2_error'])}`\n")
    text.append(f"- SNR dB: `{fmt_float(total_metrics['snr_db'])}`\n")
    text.append(f"- RMSE: `{fmt_float(total_metrics['rmse'])}`\n")
    text.append(f"- MAE: `{fmt_float(total_metrics['mae'])}`\n")
    text.append(f"- Max absolute error: `{fmt_float(total_metrics['max_abs_error'])}`\n")
    text.append("\n")
    text.append("Higher relative L2 error and lower SNR identify tensors that were more negatively affected by quantization.\n")
    text.append("\n")
    text.append(metric_reference_markdown())
    text.append("## Worst Tensors By Relative L2 Error\n\n")
    text.append(
        markdown_table(
            worst_tensors,
            [
                ("tensor", "name"),
                ("type", "quant_type"),
                ("elements", "elements"),
                ("rel_l2", "relative_l2_error"),
                ("snr_db", "snr_db"),
                ("rmse", "rmse"),
                ("mae", "mae"),
                ("max_abs", "max_abs_error"),
            ],
            args.top,
        )
    )
    text.append("\n## Lowest SNR Tensors\n\n")
    text.append(
        markdown_table(
            lowest_snr,
            [
                ("tensor", "name"),
                ("type", "quant_type"),
                ("elements", "elements"),
                ("rel_l2", "relative_l2_error"),
                ("snr_db", "snr_db"),
                ("rmse", "rmse"),
                ("mae", "mae"),
            ],
            args.top,
        )
    )
    text.append("\n## Worst Layers\n\n")
    text.append(
        markdown_table(
            layer_rows,
            [
                ("layer", "layer"),
                ("elements", "elements"),
                ("rel_l2", "relative_l2_error"),
                ("snr_db", "snr_db"),
                ("rmse", "rmse"),
                ("mae", "mae"),
            ],
            args.top,
        )
    )
    text.append("\n## Worst Sublayers Across Blocks\n\n")
    text.append(
        markdown_table(
            sublayer_rows,
            [
                ("sublayer", "sublayer"),
                ("elements", "elements"),
                ("rel_l2", "relative_l2_error"),
                ("snr_db", "snr_db"),
                ("rmse", "rmse"),
                ("mae", "mae"),
            ],
            args.top,
        )
    )
    if skipped:
        text.append("\n## Skipped\n\n")
        text.append(markdown_table(skipped, [("tensor", "name"), ("reason", "reason")], len(skipped)))

    path.write_text("".join(text), encoding="utf-8")


TENSOR_CSV_FIELDS = [
    "name",
    "layer",
    "layer_index",
    "sublayer",
    "shape",
    "elements",
    "ref_type",
    "quant_type",
    "ref_bytes",
    "quant_bytes",
    "compression_ratio",
    "mean_ref",
    "mean_quant",
    "mean_error",
    "mean_abs_ref",
    "mean_abs_quant",
    "mae",
    "mean_abs_relative_error",
    "rmse",
    "rms_ref",
    "relative_l2_error",
    "cosine_similarity",
    "snr_db",
    "max_abs_error",
    "max_abs_error_index",
    "ref_at_max_abs_error",
    "quant_at_max_abs_error",
]

GROUP_CSV_FIELDS = [
    "layer",
    "sublayer",
    "elements",
    "ref_bytes",
    "quant_bytes",
    "compression_ratio",
    "mean_ref",
    "mean_quant",
    "mean_error",
    "mean_abs_ref",
    "mean_abs_quant",
    "mae",
    "mean_abs_relative_error",
    "rmse",
    "rms_ref",
    "relative_l2_error",
    "cosine_similarity",
    "snr_db",
    "max_abs_error",
    "max_abs_error_index",
    "ref_at_max_abs_error",
    "quant_at_max_abs_error",
]


def default_path(name: str) -> Path:
    path = Path(name)
    return path if path.exists() else Path.cwd() / name


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare a reference GGUF against a quantized GGUF by dequantizing "
            "Q8_0 tensors and reporting tensor/layer/sublayer error metrics."
        )
    )
    parser.add_argument(
        "--reference",
        "--base",
        "--native",
        default="Qwen3.5-2B-BF16.gguf",
        type=Path,
        help="Reference/native GGUF, usually BF16 or F16.",
    )
    parser.add_argument(
        "--quant",
        "--candidate",
        default="Qwen3.5-2B-Q8_0.gguf",
        type=Path,
        help="Quantized GGUF to compare against the reference.",
    )
    parser.add_argument(
        "--out-dir",
        default=Path("quant_compare_report"),
        type=Path,
        help="Directory for CSV, JSON, and Markdown reports.",
    )
    parser.add_argument(
        "--chunk-blocks",
        default=262_144,
        type=int,
        help="Q8_0 blocks per streaming chunk. 262144 blocks is about 8.4M values.",
    )
    parser.add_argument(
        "--chunk-values",
        default=8_388_608,
        type=int,
        help="Dense tensor values per streaming chunk.",
    )
    parser.add_argument("--include", help="Only compare tensor names matching this regex.")
    parser.add_argument("--exclude", help="Skip tensor names matching this regex.")
    parser.add_argument("--top", default=25, type=int, help="Rows shown in Markdown top lists.")
    parser.add_argument("--max-tensors", type=int, help="Debug/validation limit after filtering.")
    parser.add_argument("--no-progress", action="store_true", help="Disable tqdm progress output.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    args.reference = default_path(str(args.reference))
    args.quant = default_path(str(args.quant))

    if args.chunk_blocks <= 0:
        raise ValueError("--chunk-blocks must be positive")
    if args.chunk_values <= 0:
        raise ValueError("--chunk-values must be positive")

    include = re.compile(args.include) if args.include else None
    exclude = re.compile(args.exclude) if args.exclude else None

    start_time = time.perf_counter()
    ref = parse_gguf(args.reference)
    quant = parse_gguf(args.quant)
    ref_by_name = ref.tensors_by_name
    quant_by_name = quant.tensors_by_name

    selected_names: list[str] = []
    skipped: list[dict[str, Any]] = []
    for tensor in ref.tensors:
        name = tensor.name
        if not passes_filters(name, include, exclude):
            continue
        if name not in quant_by_name:
            skipped.append({"name": name, "reason": "missing from candidate"})
            continue
        quant_tensor = quant_by_name[name]
        if tensor.shape != quant_tensor.shape:
            skipped.append(
                {
                    "name": name,
                    "reason": f"shape mismatch {tensor.shape_text} vs {quant_tensor.shape_text}",
                }
            )
            continue
        if tensor.ggml_type not in SUPPORTED_TYPES:
            skipped.append({"name": name, "reason": f"unsupported reference type {tensor.type_name}"})
            continue
        if quant_tensor.ggml_type not in SUPPORTED_TYPES:
            skipped.append({"name": name, "reason": f"unsupported candidate type {quant_tensor.type_name}"})
            continue
        selected_names.append(name)

    for name in quant_by_name:
        if name not in ref_by_name and passes_filters(name, include, exclude):
            skipped.append({"name": name, "reason": "missing from reference"})

    if args.max_tensors is not None:
        selected_names = selected_names[: args.max_tensors]

    if not selected_names:
        raise ValueError("no comparable tensors selected")

    total_elements = sum(ref_by_name[name].n_elements for name in selected_names)
    progress = None
    if not args.no_progress and tqdm is not None:
        progress = tqdm(total=total_elements, unit="el", unit_scale=True, desc="Comparing")

    tensor_rows: list[dict[str, Any]] = []
    ref_handle = quant_handle = None
    ref_mm = quant_mm = None
    try:
        ref_handle, ref_mm = open_mmap(ref.path)
        quant_handle, quant_mm = open_mmap(quant.path)
        for name in selected_names:
            ref_tensor = ref_by_name[name]
            quant_tensor = quant_by_name[name]
            stats = compare_tensor(
                ref_mm,
                quant_mm,
                ref_tensor,
                quant_tensor,
                args.chunk_blocks,
                args.chunk_values,
                progress,
            )
            row = make_tensor_row(ref_tensor, quant_tensor, stats)
            row.update(stats_private_fields(stats))
            tensor_rows.append(row)
    finally:
        if progress is not None:
            progress.close()
        if ref_mm is not None:
            try:
                ref_mm.close()
            except Exception:
                pass
        if quant_mm is not None:
            try:
                quant_mm.close()
            except Exception:
                pass
        if ref_handle is not None:
            ref_handle.close()
        if quant_handle is not None:
            quant_handle.close()

    tensor_rows.sort(key=lambda r: (-float(r["relative_l2_error"]), str(r["name"])))
    layer_rows = aggregate_rows(tensor_rows, "layer")
    sublayer_rows = aggregate_rows(tensor_rows, "sublayer")

    elapsed_s = time.perf_counter() - start_time
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    public_tensor_rows = [public_row(row) for row in tensor_rows]
    write_csv(out_dir / "tensor_metrics.csv", public_tensor_rows, TENSOR_CSV_FIELDS)
    write_csv(out_dir / "layer_metrics.csv", layer_rows, GROUP_CSV_FIELDS)
    write_csv(out_dir / "sublayer_metrics.csv", sublayer_rows, GROUP_CSV_FIELDS)

    summary = {
        "reference": ref.path,
        "candidate": quant.path,
        "reference_metadata": ref.metadata,
        "candidate_metadata": quant.metadata,
        "compared_tensors": len(tensor_rows),
        "compared_elements": sum(int(row["elements"]) for row in tensor_rows),
        "elapsed_seconds": elapsed_s,
        "tensor_metrics": public_tensor_rows,
        "layer_metrics": layer_rows,
        "sublayer_metrics": sublayer_rows,
        "skipped": skipped,
    }
    (out_dir / "metrics.json").write_text(
        json.dumps(json_safe(summary), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    write_markdown_report(
        out_dir / "report.md",
        args,
        ref,
        quant,
        tensor_rows,
        layer_rows,
        sublayer_rows,
        skipped,
        elapsed_s,
    )

    print(f"Compared {len(tensor_rows)} tensors ({summary['compared_elements']:,} elements)")
    print(f"Wrote {out_dir / 'report.md'}")
    print(f"Wrote {out_dir / 'tensor_metrics.csv'}")
    print(f"Wrote {out_dir / 'layer_metrics.csv'}")
    print(f"Wrote {out_dir / 'sublayer_metrics.csv'}")
    print(f"Wrote {out_dir / 'metrics.json'}")
    if skipped:
        print(f"Skipped {len(skipped)} tensors; see report.md")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted", file=sys.stderr)
        raise SystemExit(130)
