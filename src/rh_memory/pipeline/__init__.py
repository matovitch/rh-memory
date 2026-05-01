from .adapters import surrogate_training_adapter
from .config import PipelineConfig
from .primitives_harmonic import harmonic_raw_batch
from .primitives_permutation import build_grouped_permutation, gather_permuted_stream, unpermute_from_permuted
from .primitives_targets import surrogate_teacher_bucket_slot_indices
from .primitives_tokens import (
    decoder_tokens_from_surrogate_logits_soft,
    normalized_entropy,
    reshape_permuted_to_bucket_tokens,
    scalar_dib_table,
)
from .stage_harmonic import harmonic_stage
from .stage_surrogate import surrogate_stage
from .types import HarmonicSample, SurrogateInferenceSample
from .utils import iter_map, iter_take

__all__ = [
    "PipelineConfig",
    "harmonic_stage",
    "surrogate_stage",
    "HarmonicSample",
    "SurrogateInferenceSample",
    "surrogate_training_adapter",
    "harmonic_raw_batch",
    "build_grouped_permutation",
    "gather_permuted_stream",
    "unpermute_from_permuted",
    "reshape_permuted_to_bucket_tokens",
    "normalized_entropy",
    "scalar_dib_table",
    "decoder_tokens_from_surrogate_logits_soft",
    "surrogate_teacher_bucket_slot_indices",
    "iter_take",
    "iter_map",
]
