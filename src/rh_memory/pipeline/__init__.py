from .adapters import (
    reconstructor_training_adapter,
    surrogate_training_adapter,
)
from .config import PipelineConfig
from .primitives_harmonic import harmonic_raw_batch
from .primitives_permutation import build_grouped_permutation, gather_permuted_stream, unpermute_from_permuted
from .primitives_targets import surrogate_teacher_bucket_slot_indices
from .primitives_tokens import (
    decoder_tokens_from_surrogate_logits_soft,
    normalized_entropy,
    reconstructor_tokens_from_decoder_logits_hard,
    reconstructor_tokens_from_decoder_logits_soft,
    reshape_permuted_to_bucket_tokens,
    scalar_dib_table,
)
from .stage_decoder import decoder_stage
from .stage_harmonic import harmonic_stage
from .stage_surrogate import surrogate_stage
from .types import DecoderInferenceSample, HarmonicSample, SurrogateInferenceSample
from .utils import iter_map, iter_take

__all__ = [
    "PipelineConfig",
    "harmonic_stage",
    "surrogate_stage",
    "decoder_stage",
    "HarmonicSample",
    "SurrogateInferenceSample",
    "DecoderInferenceSample",
    "surrogate_training_adapter",
    "reconstructor_training_adapter",
    "harmonic_raw_batch",
    "build_grouped_permutation",
    "gather_permuted_stream",
    "unpermute_from_permuted",
    "reshape_permuted_to_bucket_tokens",
    "normalized_entropy",
    "scalar_dib_table",
    "decoder_tokens_from_surrogate_logits_soft",
    "reconstructor_tokens_from_decoder_logits_hard",
    "reconstructor_tokens_from_decoder_logits_soft",
    "surrogate_teacher_bucket_slot_indices",
    "iter_take",
    "iter_map",
]