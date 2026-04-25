from .adapters import (
    decoder_training_adapter,
    reconstructor_training_adapter,
    surrogate_training_adapter,
)
from .primitives_harmonic import harmonic_raw_batch
from .primitives_permutation import build_grouped_permutation, gather_permuted_stream, unpermute_from_permuted
from .primitives_targets import decoder_targets_from_j_star, surrogate_teacher_bucket_slot_targets
from .primitives_tokens import (
    reconstructor_tokens_from_decoder_logits,
    reshape_permuted_to_bucket_tokens,
    surrogate_bucket_tokens_from_logits,
)
from .stage_decoder import decoder_stage
from .stage_harmonic import harmonic_stage
from .stage_surrogate import surrogate_stage
from .types import DecoderSample, HarmonicSample, SurrogateSample
from .utils import iter_map, iter_take, worker_init_fn

__all__ = [
    "harmonic_stage",
    "surrogate_stage",
    "decoder_stage",
    "HarmonicSample",
    "SurrogateSample",
    "DecoderSample",
    "surrogate_training_adapter",
    "decoder_training_adapter",
    "reconstructor_training_adapter",
    "harmonic_raw_batch",
    "build_grouped_permutation",
    "gather_permuted_stream",
    "unpermute_from_permuted",
    "reshape_permuted_to_bucket_tokens",
    "surrogate_bucket_tokens_from_logits",
    "reconstructor_tokens_from_decoder_logits",
    "surrogate_teacher_bucket_slot_targets",
    "decoder_targets_from_j_star",
    "iter_take",
    "iter_map",
    "worker_init_fn",
]
