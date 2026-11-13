"""Image classifiers for pseudo labels: the protocol, image preparation and the ONNX adapter.

:class:`ImageClassifier` is what :mod:`tools.repair.pseudo_labels` depends on;
tests pass a fake. :class:`OnnxImageClassifier` runs a
:class:`core.class_labels.PseudoModel` with onnxruntime on the CPU; the model
file is fetched from the Hugging Face hub at its pinned revision (the hub
cache keeps it). :func:`prepare_images` turns encoded images into the
models' input pixels (RGB, nearest-neighbour resize to a square, uint8);
:func:`to_model_input` scales a batch to 0..1 floats.
"""

from __future__ import annotations

import io
from typing import Any, Protocol

import numpy as np
from PIL import Image

from core.class_labels import PseudoModel
from core.lazy_imports import load_hf_hub_download, load_onnxruntime

__all__ = [
    "ImageClassifier",
    "OnnxImageClassifier",
    "prepare_images",
    "to_model_input",
]

_SUM_TOLERANCE = 1e-3


class ImageClassifier(Protocol):
    """Class probabilities of prepared images."""

    @property
    def input_size(self) -> int:
        """Side of the square input in pixels."""
        ...

    def predict(self, batch: np.ndarray) -> np.ndarray:
        """``(n, classes)`` probabilities of an ``(n, size, size, 3)`` uint8 batch."""
        ...


def prepare_images(payloads: list[bytes], sizes: tuple[int, ...]) -> list[list[np.ndarray]]:
    """Per size, the ``(size, size, 3)`` uint8 RGB pixels of every payload (each image
    decoded once, nearest-neighbour resize)."""
    out: list[list[np.ndarray]] = [[] for _ in sizes]
    for payload in payloads:
        with Image.open(io.BytesIO(payload)) as img:
            rgb = img.convert("RGB")
        for i, size in enumerate(sizes):
            out[i].append(np.asarray(rgb.resize((size, size), Image.Resampling.NEAREST),
                                     dtype=np.uint8))
    return out


def to_model_input(batch: np.ndarray) -> np.ndarray:
    """A uint8 NHWC batch as float32 scaled to 0..1."""
    return batch.astype(np.float32) / np.float32(255.0)


class OnnxImageClassifier:
    """A pinned ONNX model from the Hugging Face hub, run with onnxruntime on the CPU."""

    def __init__(self, model: PseudoModel, threads: int = 0) -> None:
        onnxruntime = load_onnxruntime()
        path = load_hf_hub_download()(model.repo_id, model.filename, revision=model.revision)
        options = onnxruntime.SessionOptions()
        options.intra_op_num_threads = max(0, threads)
        self._session: Any = onnxruntime.InferenceSession(
            path, sess_options=options, providers=["CPUExecutionProvider"])
        self._input = self._session.get_inputs()[0].name
        self.model = model

    @property
    def input_size(self) -> int:
        """Side of the model's square input in pixels."""
        return self.model.input_size

    def predict(self, batch: np.ndarray) -> np.ndarray:
        """Softmax probabilities; ``ValueError`` when the output is not a distribution."""
        probs = np.asarray(self._session.run(None, {self._input: to_model_input(batch)})[0],
                           dtype=np.float64)
        classes = len(self.model.class_names)
        if probs.shape != (batch.shape[0], classes) \
                or not np.allclose(probs.sum(axis=1), 1.0, atol=_SUM_TOLERANCE):
            raise ValueError(f"{self.model.name}: output {probs.shape} is not a probability"
                             f" distribution over {classes} classes")
        return probs
