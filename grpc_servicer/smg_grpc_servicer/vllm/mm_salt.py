"""Multimodal identity cache salt for tensor-stripped (PD decode) legs.

The PD router strips multimodal tensors from the decode leg (the KV arrives
via the P/D transfer), keeping only the per-image content hashes. vLLM folds
multimodal identity into its prefix-cache block hashes exclusively through
``mm_features``, which cannot be built without the tensors — so on this leg
the identity rides ``cache_salt``, which enters the chained block hash at
block 0. The salt is deterministic per image content: reuse of the same image
still hits the decode-side prefix cache, while two different images behind
the same text prefix no longer alias onto each other's KV.

Must stay in sync with ``mm_identity_cache_salt`` in
``model_gateway/src/routers/grpc/zmq_multimodal.rs`` (the direct-ZMQ path).
"""

from collections.abc import Sequence


def mm_identity_cache_salt(mm_hashes: Sequence[str]) -> str | None:
    """Fold per-image content hashes into a deterministic cache salt."""
    if not mm_hashes:
        return None
    return "mm:" + ",".join(mm_hashes)
