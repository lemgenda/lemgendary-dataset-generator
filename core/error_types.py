"""Single source of truth for the expected failure types of compiler stages.

Handlers catch these tuples instead of ``Exception`` so that programming
errors (``TypeError``, ``AttributeError``, ``NameError``, ...) are never
swallowed by a per-sample or per-model recovery path.
"""

from PIL import Image

# Decoding, validating, transforming and encoding one image sample.
# PIL reports unreadable data as OSError (UnidentifiedImageError) or
# ValueError; oversized images raise DecompressionBombError; torch and numpy
# kernels raise RuntimeError / ValueError / IndexError; malformed annotation
# records raise KeyError / IndexError.
SAMPLE_PROCESSING_ERRORS: tuple[type[BaseException], ...] = (
    OSError,
    ValueError,
    KeyError,
    IndexError,
    RuntimeError,
    Image.DecompressionBombError,
)

# Loading an optional neural model (quality sentry, captioner, CLIP, labeler):
# missing package, missing / unreadable weights, CUDA or checkpoint mismatch.
MODEL_LOAD_ERRORS: tuple[type[BaseException], ...] = (
    ImportError,
    OSError,
    RuntimeError,
    ValueError,
)

# Reading a parquet shard with pandas / pyarrow: pyarrow.ArrowInvalid derives
# from ValueError and ArrowIOError from OSError. ImportError (pyarrow missing)
# is deliberately excluded so a broken environment is not mistaken for a
# corrupt shard.
PARQUET_READ_ERRORS: tuple[type[BaseException], ...] = (OSError, ValueError)

# Regenerating a manifold's documentation bundle (README, YAML / JSON
# metadata, cover image): file-system errors, malformed source metadata and
# unreadable or oversized images.
DOC_GENERATION_ERRORS: tuple[type[BaseException], ...] = (
    OSError,
    ValueError,
    KeyError,
    Image.DecompressionBombError,
)
