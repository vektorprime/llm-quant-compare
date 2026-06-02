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
import heapq
import json
import math
import mmap
import os
import re
import struct
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
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
    source_path: Path = field(default_factory=Path)

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


@dataclass
class GGUFCollection:
    label: str
    files: list[GGUFFile]

    @property
    def paths(self) -> list[Path]:
        return [file.path for file in self.files]

    @property
    def path_text(self) -> str:
        return ", ".join(str(path) for path in self.paths)

    @property
    def metadata(self) -> dict[str, Any]:
        return self.files[0].metadata if self.files else {}

    @property
    def tensors(self) -> list[TensorInfo]:
        return [tensor for file in self.files for tensor in file.tensors]

    @property
    def tensors_by_name(self) -> dict[str, TensorInfo]:
        by_name: dict[str, TensorInfo] = {}
        for tensor in self.tensors:
            if tensor.name in by_name:
                first = by_name[tensor.name]
                raise ValueError(
                    f"{self.label}: duplicate tensor name {tensor.name!r} in "
                    f"{first.source_path} and {tensor.source_path}"
                )
            by_name[tensor.name] = tensor
        return by_name


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
            tensors.append(TensorInfo(name, shape, ggml_type, relative_offset, source_path=path))

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


def build_collection(label: str, paths: list[Path]) -> GGUFCollection:
    if not paths:
        raise ValueError(f"{label}: no GGUF paths provided")
    files = [parse_gguf(path) for path in paths]
    collection = GGUFCollection(label=label, files=files)
    # Force duplicate detection early, before the expensive comparison starts.
    _ = collection.tensors_by_name
    return collection


def open_mmap(path: Path) -> tuple[Any, mmap.mmap]:
    f = path.open("rb")
    try:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    except Exception:
        f.close()
        raise
    return f, mm


def open_mmaps(paths: list[Path]) -> tuple[list[Any], list[mmap.mmap]]:
    handles: list[Any] = []
    maps: list[mmap.mmap] = []
    try:
        for path in paths:
            handle, mm = open_mmap(path)
            handles.append(handle)
            maps.append(mm)
    except Exception:
        close_mmaps(handles, maps)
        raise
    return handles, maps


def close_mmaps(handles: list[Any], maps: list[mmap.mmap]) -> None:
    for mm in maps:
        try:
            mm.close()
        except Exception:
            pass
    for handle in handles:
        try:
            handle.close()
        except Exception:
            pass


def dequant_q8_0(mm: mmap.mmap, tensor: TensorInfo, block_start: int, block_count: int) -> np.ndarray:
    byte_offset = tensor.data_offset + block_start * Q8_0_DTYPE.itemsize
    blocks = np.frombuffer(mm, dtype=Q8_0_DTYPE, count=block_count, offset=byte_offset)
    values = blocks["qs"].astype(np.float32)
    values *= blocks["d"].astype(np.float32).reshape(-1, 1)
    return values.reshape(-1)


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
        abs_error = np.abs(error)

        self.n += int(ref.size)
        self.sum_ref += float(np.sum(ref, dtype=np.float64))
        self.sum_quant += float(np.sum(quant, dtype=np.float64))
        self.sum_abs_ref += float(np.sum(np.abs(ref), dtype=np.float64))
        self.sum_abs_quant += float(np.sum(np.abs(quant), dtype=np.float64))
        self.sum_error += float(np.sum(error, dtype=np.float64))
        self.sum_abs_error += float(np.sum(abs_error, dtype=np.float64))
        self.sum_squared_error += float(np.einsum("i,i->", error, error, dtype=np.float64))
        self.sum_squared_ref += float(np.einsum("i,i->", ref, ref, dtype=np.float64))
        self.sum_squared_quant += float(np.einsum("i,i->", quant, quant, dtype=np.float64))
        self.dot += float(np.einsum("i,i->", ref, quant, dtype=np.float64))

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


@dataclass
class BlockOutlierInfo:
    """Per-Q8_0-block statistics used to detect outlier-driven scale divergence."""
    tensor_name: str
    block_index: int          # 0-based block index within the tensor
    scale: float              # the float16 scale d (dequantized to float32)
    max_abs_ref: float        # max |ref| value in this 32-value block
    rms_ref: float            # RMS of ref values in this block
    max_abs_error: float      # max |quant - ref| in this block
    outlier_score: float      # max_abs_ref / rms_ref  (higher = more extreme outlier)

    @property
    def global_element_index(self) -> int:
        """Flattened element index where this block starts."""
        return self.block_index * Q8_0_BLOCK_SIZE


def split_layer_sublayer(name: str) -> tuple[str, str, int | None]:
    match = BLOCK_NAME_RE.match(name)
    if match:
        layer_index = int(match.group(1))
        return f"blk.{layer_index}", match.group(2), layer_index
    return "global", name, None


@dataclass
class SublayerBlockOutlierAccumulator:
    total_q8_blocks: int = 0
    sum_outlier_score: float = 0.0
    sum_max_abs_error: float = 0.0
    best: BlockOutlierInfo | None = None

    def update(
        self,
        count: int,
        score_sum: float,
        max_abs_error_sum: float,
        best: BlockOutlierInfo,
    ) -> None:
        self.total_q8_blocks += count
        self.sum_outlier_score += score_sum
        self.sum_max_abs_error += max_abs_error_sum
        if self.best is None or best.outlier_score > self.best.outlier_score:
            self.best = best


class Q8BlockOutlierAnalyzer:
    """Streaming top-N and grouped Q8_0 block outlier analysis.

    The original implementation kept one Python object for every Q8_0 block and
    sorted them at the end. Large GGUFs can contain tens of millions of blocks,
    so that made this phase both CPU- and memory-heavy. This keeps only the
    requested top-N blocks while still accumulating exact per-sublayer totals.
    """

    def __init__(self, top: int) -> None:
        self.top = max(0, int(top))
        self.total_blocks = 0
        self._counter = 0
        self._top_heap: list[tuple[float, int, BlockOutlierInfo]] = []
        self._sublayers: dict[tuple[str, str], SublayerBlockOutlierAccumulator] = {}

    def update(
        self,
        ref_tensor: TensorInfo,
        quant_tensor: TensorInfo,
        ref: np.ndarray,
        quant: np.ndarray,
        quant_mm: mmap.mmap,
        block_start: int,
        block_count: int,
    ) -> None:
        if block_count <= 0:
            return

        byte_offset = quant_tensor.data_offset + block_start * Q8_0_DTYPE.itemsize
        raw_blocks = np.frombuffer(quant_mm, dtype=Q8_0_DTYPE, count=block_count, offset=byte_offset)
        scales = raw_blocks["d"].astype(np.float32)

        ref_blocks = ref.reshape(block_count, Q8_0_BLOCK_SIZE)
        quant_blocks = quant.reshape(block_count, Q8_0_BLOCK_SIZE)
        max_abs_ref = np.max(np.abs(ref_blocks), axis=1)
        ref_sq_sum = np.einsum("ij,ij->i", ref_blocks, ref_blocks, dtype=np.float64)
        rms_ref = np.sqrt(ref_sq_sum / Q8_0_BLOCK_SIZE)
        error = np.subtract(quant_blocks, ref_blocks, dtype=np.float32)
        max_abs_error = np.max(np.abs(error), axis=1)
        scores = max_abs_ref / np.maximum(rms_ref, 1e-30)

        self.total_blocks += block_count

        best_index = int(np.argmax(scores))
        best = self._make_block(
            ref_tensor.name,
            block_start + best_index,
            scales,
            max_abs_ref,
            rms_ref,
            max_abs_error,
            scores,
            best_index,
        )
        layer, sublayer, _ = split_layer_sublayer(ref_tensor.name)
        key = (layer, sublayer)
        self._sublayers.setdefault(key, SublayerBlockOutlierAccumulator()).update(
            block_count,
            float(np.sum(scores, dtype=np.float64)),
            float(np.sum(max_abs_error, dtype=np.float64)),
            best,
        )

        if self.top <= 0:
            return

        candidate_count = min(self.top, block_count)
        if candidate_count == block_count:
            candidate_indexes = np.arange(block_count)
        else:
            candidate_indexes = np.argpartition(scores, -candidate_count)[-candidate_count:]

        for raw_index in candidate_indexes:
            i = int(raw_index)
            block = self._make_block(
                ref_tensor.name,
                block_start + i,
                scales,
                max_abs_ref,
                rms_ref,
                max_abs_error,
                scores,
                i,
            )
            item = (block.outlier_score, self._counter, block)
            self._counter += 1
            if len(self._top_heap) < self.top:
                heapq.heappush(self._top_heap, item)
            elif item[0] > self._top_heap[0][0]:
                heapq.heapreplace(self._top_heap, item)

    @staticmethod
    def _make_block(
        tensor_name: str,
        block_index: int,
        scales: np.ndarray,
        max_abs_ref: np.ndarray,
        rms_ref: np.ndarray,
        max_abs_error: np.ndarray,
        scores: np.ndarray,
        i: int,
    ) -> BlockOutlierInfo:
        return BlockOutlierInfo(
            tensor_name=tensor_name,
            block_index=block_index,
            scale=float(scales[i]),
            max_abs_ref=float(max_abs_ref[i]),
            rms_ref=float(rms_ref[i]),
            max_abs_error=float(max_abs_error[i]),
            outlier_score=float(scores[i]),
        )

    def worst_block_rows(self) -> list[dict[str, Any]]:
        blocks = [item[2] for item in self._top_heap]
        blocks.sort(key=lambda b: -b.outlier_score)
        return [_block_outlier_to_row(block) for block in blocks]

    def sublayer_summary_rows(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for (layer, sublayer), acc in self._sublayers.items():
            if acc.total_q8_blocks == 0 or acc.best is None:
                continue
            best = acc.best
            rows.append({
                "layer": layer,
                "sublayer": sublayer,
                "tensor_name": best.tensor_name,
                "total_q8_blocks": acc.total_q8_blocks,
                "mean_outlier_score": acc.sum_outlier_score / acc.total_q8_blocks,
                "max_outlier_score": best.outlier_score,
                "worst_block_index": best.block_index,
                "worst_block_global_index": best.global_element_index,
                "worst_block_scale": best.scale,
                "worst_block_max_abs_ref": best.max_abs_ref,
                "worst_block_rms_ref": best.rms_ref,
                "worst_block_max_abs_error": best.max_abs_error,
                "mean_max_abs_error": acc.sum_max_abs_error / acc.total_q8_blocks,
            })
        rows.sort(key=lambda r: -float(r["max_outlier_score"]))
        return rows


def compare_tensor(
    ref_mm: mmap.mmap,
    quant_mm: mmap.mmap,
    ref_tensor: TensorInfo,
    quant_tensor: TensorInfo,
    chunk_blocks: int,
    chunk_values: int,
    progress: Any | None,
    collect_block_outliers: bool = False,
    block_outlier_analyzer: Q8BlockOutlierAnalyzer | None = None,
) -> tuple[RunningStats, list[BlockOutlierInfo]]:
    """Compare a tensor and optionally collect per-block outlier info for Q8_0 tensors.

    Returns (stats, block_outliers). For the CLI path, block_outlier_analyzer
    streams Q8_0 block analysis without storing every block. If callers use the
    older collect_block_outliers=True API without an analyzer, the full list is
    still returned for compatibility.
    """
    stats = RunningStats()
    block_outliers: list[BlockOutlierInfo] = []
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

            if collect_block_outliers and quant_tensor.ggml_type == GGML_Q8_0:
                if block_outlier_analyzer is not None:
                    block_outlier_analyzer.update(
                        ref_tensor, quant_tensor, ref, quant, quant_mm, block_start, block_count
                    )
                else:
                    _collect_q8_0_block_outliers(
                        block_outliers, ref_tensor, quant_tensor,
                        ref, quant, quant_mm, block_start, block_count,
                    )

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

    return stats, block_outliers


def _collect_q8_0_block_outliers(
    block_outliers: list[BlockOutlierInfo],
    ref_tensor: TensorInfo,
    quant_tensor: TensorInfo,
    ref: np.ndarray,
    quant: np.ndarray,
    quant_mm: mmap.mmap,
    block_start: int,
    block_count: int,
) -> None:
    """Compatibility path that materializes per-block outlier objects in block order."""
    byte_offset = quant_tensor.data_offset + block_start * Q8_0_DTYPE.itemsize
    raw_blocks = np.frombuffer(quant_mm, dtype=Q8_0_DTYPE, count=block_count, offset=byte_offset)
    scales = raw_blocks["d"].astype(np.float32)

    ref_blocks = ref.reshape(block_count, Q8_0_BLOCK_SIZE)
    quant_blocks = quant.reshape(block_count, Q8_0_BLOCK_SIZE)
    max_abs_ref = np.max(np.abs(ref_blocks), axis=1)
    ref_sq_sum = np.einsum("ij,ij->i", ref_blocks, ref_blocks, dtype=np.float64)
    rms_ref = np.sqrt(ref_sq_sum / Q8_0_BLOCK_SIZE)
    error = np.subtract(quant_blocks, ref_blocks, dtype=np.float32)
    max_abs_error = np.max(np.abs(error), axis=1)
    scores = max_abs_ref / np.maximum(rms_ref, 1e-30)

    for i in range(block_count):
        block_outliers.append(BlockOutlierInfo(
            tensor_name=ref_tensor.name,
            block_index=block_start + i,
            scale=float(scales[i]),
            max_abs_ref=float(max_abs_ref[i]),
            rms_ref=float(rms_ref[i]),
            max_abs_error=float(max_abs_error[i]),
            outlier_score=float(scores[i]),
        ))


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
        "ref_file": str(ref_tensor.source_path),
        "quant_file": str(quant_tensor.source_path),
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


def analyze_block_outliers(
    analyzer: Q8BlockOutlierAnalyzer,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return Q8_0 block outlier outputs from the streaming analyzer."""
    if analyzer.total_blocks == 0:
        return [], []
    return analyzer.worst_block_rows(), analyzer.sublayer_summary_rows()


def _block_outlier_to_row(b: BlockOutlierInfo) -> dict[str, Any]:
    layer, sublayer, layer_index = split_layer_sublayer(b.tensor_name)
    return {
        "tensor_name": b.tensor_name,
        "layer": layer,
        "layer_index": layer_index,
        "sublayer": sublayer,
        "block_index": b.block_index,
        "global_element_index": b.global_element_index,
        "scale": b.scale,
        "max_abs_ref": b.max_abs_ref,
        "rms_ref": b.rms_ref,
        "max_abs_error": b.max_abs_error,
        "outlier_score": b.outlier_score,
    }


BLOCK_OUTLIER_CSV_FIELDS = [
    "tensor_name",
    "layer",
    "layer_index",
    "sublayer",
    "block_index",
    "global_element_index",
    "scale",
    "max_abs_ref",
    "rms_ref",
    "max_abs_error",
    "outlier_score",
]

SUBLAYER_BLOCK_OUTLIER_CSV_FIELDS = [
    "layer",
    "sublayer",
    "tensor_name",
    "total_q8_blocks",
    "mean_outlier_score",
    "max_outlier_score",
    "worst_block_index",
    "worst_block_global_index",
    "worst_block_scale",
    "worst_block_max_abs_ref",
    "worst_block_rms_ref",
    "worst_block_max_abs_error",
    "mean_max_abs_error",
]


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


def has_measured_error(row: dict[str, Any]) -> bool:
    for key in ("relative_l2_error", "rmse", "mae", "max_abs_error"):
        value = float(row.get(key, 0.0))
        if math.isnan(value):
            return True
        if value != 0.0:
            return True
    return False


def type_counts(gguf: GGUFFile | GGUFCollection) -> dict[str, int]:
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

"""


def write_markdown_report(
    path: Path,
    args: argparse.Namespace,
    ref: GGUFCollection,
    quant: GGUFCollection,
    tensor_rows: list[dict[str, Any]],
    layer_rows: list[dict[str, Any]],
    sublayer_rows: list[dict[str, Any]],
    skipped: list[dict[str, Any]],
    elapsed_s: float,
    worst_block_outliers: list[dict[str, Any]] | None = None,
    sublayer_block_outlier_rows: list[dict[str, Any]] | None = None,
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

    report_tensor_rows = tensor_rows if args.show_zero_error else [
        row for row in tensor_rows if has_measured_error(row)
    ]
    report_layer_rows = layer_rows if args.show_zero_error else [
        row for row in layer_rows if has_measured_error(row)
    ]
    report_sublayer_rows = sublayer_rows if args.show_zero_error else [
        row for row in sublayer_rows if has_measured_error(row)
    ]
    hidden_zero_error_rows = len(tensor_rows) - len(report_tensor_rows)

    worst_tensors = sorted(report_tensor_rows, key=lambda r: -float(r["relative_l2_error"]))
    lowest_snr = sorted(report_tensor_rows, key=lambda r: float(r["snr_db"]))

    text = []
    text.append("# GGUF Quantization Comparison\n")
    text.append(f"- Output directory: `{path.parent}`\n")
    text.append(f"- Reference: `{ref.path_text}`\n")
    text.append(f"- Candidate: `{quant.path_text}`\n")
    text.append(f"- Reference files: {len(ref.files)}\n")
    text.append(f"- Candidate files: {len(quant.files)}\n")
    text.append(f"- Compared tensors: {len(tensor_rows)}\n")
    text.append(f"- Compared elements: {total_metrics['elements']:,}\n")
    text.append(f"- Elapsed: {elapsed_s:.2f} seconds\n")
    text.append(f"- Reference tensor types: `{json.dumps(type_counts(ref), sort_keys=True)}`\n")
    text.append(f"- Candidate tensor types: `{json.dumps(type_counts(quant), sort_keys=True)}`\n")
    if hidden_zero_error_rows:
        text.append(f"- Zero-error tensor rows hidden from ranked tables: {hidden_zero_error_rows}\n")
    if skipped and not args.show_skipped:
        text.append(f"- Skipped tensor rows hidden from this report: {len(skipped)}\n")
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
            report_layer_rows,
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
            report_sublayer_rows,
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
    if worst_block_outliers and sublayer_block_outlier_rows:
        text.append("\n## Q8_0 Block Outlier Analysis\n\n")
        text.append(
            "Each Q8_0 tensor is stored as 32-value blocks that share a single float16 scale `d`. "
            "The dequantized value is `d * int8_weight`. "
            "When one or a few values in a block have much larger magnitude than the rest, "
            "the scale is forced up to accommodate them, which reduces precision for all other "
            "values in that block.\n\n"
        )
        text.append(
            "**Outlier score** = `max_abs_ref / rms_ref` for each 32-value block. "
            "A higher score means the reference values within that block have a more extreme "
            "outlier (max value is much larger than the typical/RMS value). "
            "These outlier-driven blocks are the primary cause of Q8_0 scale divergence "
            "and precision loss.\n\n"
        )
        text.append("### Worst Individual Q8_0 Blocks by Outlier Score\n\n")
        text.append(
            markdown_table(
                worst_block_outliers,
                [
                    ("tensor", "tensor_name"),
                    ("layer", "layer"),
                    ("sublayer", "sublayer"),
                    ("block idx", "block_index"),
                    ("outlier score", "outlier_score"),
                    ("scale (d)", "scale"),
                    ("max|ref|", "max_abs_ref"),
                    ("rms ref", "rms_ref"),
                    ("max|error|", "max_abs_error"),
                ],
                args.block_outlier_top if hasattr(args, 'block_outlier_top') else args.top,
            )
        )
        text.append("\n### Sublayers Most Affected by Q8_0 Block Outliers\n\n")
        text.append(
            "Aggregated across all transformer blocks. "
            "Higher `max_outlier_score` indicates sublayers where at least one Q8_0 block "
            "has a severe outlier that drives the shared scale away from the typical value range.\n\n"
        )
        text.append(
            markdown_table(
                sublayer_block_outlier_rows,
                [
                    ("sublayer", "sublayer"),
                    ("blocks", "total_q8_blocks"),
                    ("max outlier score", "max_outlier_score"),
                    ("mean outlier score", "mean_outlier_score"),
                    ("worst block idx", "worst_block_index"),
                    ("worst scale (d)", "worst_block_scale"),
                    ("worst max|ref|", "worst_block_max_abs_ref"),
                    ("worst rms ref", "worst_block_rms_ref"),
                    ("worst max|error|", "worst_block_max_abs_error"),
                ],
                args.block_outlier_top if hasattr(args, 'block_outlier_top') else args.top,
            )
        )
    if skipped and args.show_skipped:
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
    "ref_file",
    "quant_file",
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


def default_path(name: str | Path) -> Path:
    path = Path(name)
    return path if path.exists() or path.is_absolute() else Path.cwd() / path


def expand_path_args(values: list[Path], label: str) -> list[Path]:
    expanded: list[Path] = []
    for value in values:
        text = str(value)
        if any(ch in text for ch in "*?[]"):
            pattern_path = Path(text)
            if pattern_path.is_absolute():
                matches = sorted(pattern_path.parent.glob(pattern_path.name))
            else:
                matches = sorted(Path.cwd().glob(text))
            if not matches:
                raise FileNotFoundError(f"{label}: pattern matched no files: {text}")
            expanded.extend(matches)
        else:
            expanded.append(default_path(value))

    missing = [path for path in expanded if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"{label}: missing file(s): " + ", ".join(str(path) for path in missing)
        )

    return expanded


def strip_model_suffixes(stem: str) -> str:
    out = re.sub(r"-\d{5}-of-\d{5}$", "", stem)
    quant_suffix = (
        r"-(?:BF16|F16|F32|Q\d(?:_[0-9A-Z]+)+|IQ\d(?:_[0-9A-Z]+)+|"
        r"UD-[0-9A-Z_]+|I\d+|MXFP4|TQ\d_0)$"
    )
    while True:
        stripped = re.sub(quant_suffix, "", out, flags=re.IGNORECASE)
        if stripped == out:
            return out
        out = stripped


def sanitize_path_component(text: str) -> str:
    text = text.strip()
    text = re.sub(r"[<>:\"/\\|?*\x00-\x1f]+", "-", text)
    text = re.sub(r"\s+", "-", text)
    text = re.sub(r"-{2,}", "-", text)
    return text.strip("-.") or "model"


def infer_model_name(ref: GGUFCollection, quant: GGUFCollection) -> str:
    for collection in (ref, quant):
        name = collection.metadata.get("general.name")
        if isinstance(name, str) and name.strip():
            return sanitize_path_component(strip_model_suffixes(name.strip()))

    stems = [strip_model_suffixes(path.stem) for path in ref.paths + quant.paths]
    if stems:
        common = os.path.commonprefix(stems).rstrip("-. _")
        if common:
            return sanitize_path_component(common)
        return sanitize_path_component(stems[0])
    return "model"


def create_run_output_dir(base_dir: Path, model_name: str, started_at: datetime) -> Path:
    base_dir.mkdir(parents=True, exist_ok=True)
    timestamp = started_at.strftime("%Y%m%d-%H%M%S")
    stem = f"{sanitize_path_component(model_name)}-{timestamp}"

    for i in range(10_000):
        suffix = "" if i == 0 else f"-{i + 1}"
        out_dir = base_dir / f"{stem}{suffix}"
        try:
            out_dir.mkdir(parents=True, exist_ok=False)
            return out_dir
        except FileExistsError:
            continue

    raise FileExistsError(f"could not create a unique output directory under {base_dir}")


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
        default=[Path("Qwen3.5-2B-BF16.gguf")],
        nargs="+",
        type=Path,
        help=(
            "Reference/native GGUF path(s), usually BF16 or F16. "
            "Pass multiple files or a glob for split GGUFs."
        ),
    )
    parser.add_argument(
        "--quant",
        "--candidate",
        default=[Path("Qwen3.5-2B-Q8_0.gguf")],
        nargs="+",
        type=Path,
        help=(
            "Quantized GGUF path(s) to compare against the reference. "
            "Pass multiple files or a glob for split GGUFs."
        ),
    )
    parser.add_argument(
        "--out-dir",
        default=Path("quant_compare_report"),
        type=Path,
        help=(
            "Base directory for reports. Each run creates a unique "
            "model-name timestamped subdirectory here."
        ),
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
    parser.add_argument(
        "--show-zero-error",
        action="store_true",
        help="Include zero-error rows in Markdown ranked tables.",
    )
    parser.add_argument(
        "--show-skipped",
        action="store_true",
        help="Include skipped tensor details in the Markdown report.",
    )
    parser.add_argument(
        "--block-outlier-top",
        default=25,
        type=int,
        help="Rows shown in Q8_0 block outlier Markdown tables (default: 25).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    args.reference = expand_path_args(args.reference, "reference")
    args.quant = expand_path_args(args.quant, "candidate")
    run_started_at = datetime.now()

    if args.chunk_blocks <= 0:
        raise ValueError("--chunk-blocks must be positive")
    if args.chunk_values <= 0:
        raise ValueError("--chunk-values must be positive")

    include = re.compile(args.include) if args.include else None
    exclude = re.compile(args.exclude) if args.exclude else None

    start_time = time.perf_counter()
    ref = build_collection("reference", args.reference)
    quant = build_collection("candidate", args.quant)
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

    model_name = infer_model_name(ref, quant)
    out_dir = create_run_output_dir(args.out_dir, model_name, run_started_at)

    total_elements = sum(ref_by_name[name].n_elements for name in selected_names)
    progress = None
    if not args.no_progress and tqdm is not None:
        progress = tqdm(total=total_elements, unit="el", unit_scale=True, desc="Comparing")

    tensor_rows: list[dict[str, Any]] = []
    block_outlier_analyzer = Q8BlockOutlierAnalyzer(args.block_outlier_top)
    ref_handles: list[Any] = []
    quant_handles: list[Any] = []
    ref_mmaps: list[mmap.mmap] = []
    quant_mmaps: list[mmap.mmap] = []
    try:
        ref_handles, ref_mmaps = open_mmaps(ref.paths)
        quant_handles, quant_mmaps = open_mmaps(quant.paths)
        ref_mmap_by_path = dict(zip(ref.paths, ref_mmaps))
        quant_mmap_by_path = dict(zip(quant.paths, quant_mmaps))
        for name in selected_names:
            ref_tensor = ref_by_name[name]
            quant_tensor = quant_by_name[name]
            stats, block_outliers = compare_tensor(
                ref_mmap_by_path[ref_tensor.source_path],
                quant_mmap_by_path[quant_tensor.source_path],
                ref_tensor,
                quant_tensor,
                args.chunk_blocks,
                args.chunk_values,
                progress,
                collect_block_outliers=True,
                block_outlier_analyzer=block_outlier_analyzer,
            )
            row = make_tensor_row(ref_tensor, quant_tensor, stats)
            row.update(stats_private_fields(stats))
            tensor_rows.append(row)
    finally:
        if progress is not None:
            progress.close()
        close_mmaps(ref_handles, ref_mmaps)
        close_mmaps(quant_handles, quant_mmaps)

    tensor_rows.sort(key=lambda r: (-float(r["relative_l2_error"]), str(r["name"])))
    layer_rows = aggregate_rows(tensor_rows, "layer")
    sublayer_rows = aggregate_rows(tensor_rows, "sublayer")

    # Analyze Q8_0 block outliers
    worst_blocks, sublayer_block_outliers = analyze_block_outliers(block_outlier_analyzer)

    elapsed_s = time.perf_counter() - start_time

    public_tensor_rows = [public_row(row) for row in tensor_rows]
    write_csv(out_dir / "tensor_metrics.csv", public_tensor_rows, TENSOR_CSV_FIELDS)
    write_csv(out_dir / "layer_metrics.csv", layer_rows, GROUP_CSV_FIELDS)
    write_csv(out_dir / "sublayer_metrics.csv", sublayer_rows, GROUP_CSV_FIELDS)
    if worst_blocks:
        write_csv(out_dir / "block_outliers.csv", worst_blocks, BLOCK_OUTLIER_CSV_FIELDS)
    if sublayer_block_outliers:
        write_csv(
            out_dir / "sublayer_block_outliers.csv",
            sublayer_block_outliers,
            SUBLAYER_BLOCK_OUTLIER_CSV_FIELDS,
        )

    summary = {
        "model_name": model_name,
        "run_started_at": run_started_at.isoformat(timespec="seconds"),
        "output_base_dir": args.out_dir,
        "output_dir": out_dir,
        "reference": ref.paths,
        "candidate": quant.paths,
        "reference_metadata": ref.metadata,
        "candidate_metadata": quant.metadata,
        "compared_tensors": len(tensor_rows),
        "compared_elements": sum(int(row["elements"]) for row in tensor_rows),
        "elapsed_seconds": elapsed_s,
        "tensor_metrics": public_tensor_rows,
        "layer_metrics": layer_rows,
        "sublayer_metrics": sublayer_rows,
        "skipped": skipped,
        "q8_0_blocks_analyzed": block_outlier_analyzer.total_blocks,
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
        worst_block_outliers=worst_blocks if worst_blocks else None,
        sublayer_block_outlier_rows=sublayer_block_outliers if sublayer_block_outliers else None,
    )

    print(f"Compared {len(tensor_rows)} tensors ({summary['compared_elements']:,} elements)")
    print(f"Output directory: {out_dir}")
    print(f"Wrote {out_dir / 'report.md'}")
    print(f"Wrote {out_dir / 'tensor_metrics.csv'}")
    print(f"Wrote {out_dir / 'layer_metrics.csv'}")
    print(f"Wrote {out_dir / 'sublayer_metrics.csv'}")
    if worst_blocks:
        print(f"Wrote {out_dir / 'block_outliers.csv'} ({block_outlier_analyzer.total_blocks:,} Q8_0 blocks analyzed)")
    if sublayer_block_outliers:
        print(f"Wrote {out_dir / 'sublayer_block_outliers.csv'}")
    print(f"Wrote {out_dir / 'metrics.json'}")
    if skipped:
        if args.show_skipped:
            print(f"Skipped {len(skipped)} tensors; see report.md and metrics.json")
        else:
            print(
                f"Skipped {len(skipped)} tensors; details are in metrics.json "
                f"(use --show-skipped to include them in report.md)"
            )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted", file=sys.stderr)
        raise SystemExit(130)
