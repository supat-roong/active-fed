import io

import torch

from src.agent.model import ActorCritic
from src.aggregator.collect import write_initial_global_weights


class FakeMinio:
    """Minimal stand-in recording puts and answering existence checks."""

    def __init__(self, existing=None):
        self.objects: dict[str, bytes] = dict(existing or {})
        self.puts: list[str] = []

    def bucket_exists(self, bucket):
        return True

    def stat_object(self, bucket, key):
        if key not in self.objects:
            from minio.error import S3Error

            # S3Error's signature is (response, code, message, resource,
            # request_id, host_id, ...) — code must be the *second* arg so
            # that write_initial_global_weights' `e.code == "NoSuchKey"`
            # check actually matches a missing-key response.
            raise S3Error(None, "NoSuchKey", "missing", key, "rid", "hid")
        return object()

    def put_object(self, bucket, key, data, length):
        self.puts.append(key)
        self.objects[key] = data.read()

    def get_object(self, bucket, key):
        class R:
            def __init__(self, b):
                self._b = b

            def read(self):
                return self._b

        return R(self.objects[key])


def _load(client, key):
    return torch.load(io.BytesIO(client.objects[key]), map_location="cpu", weights_only=True)


def test_writes_round_zero_global_weights():
    c = FakeMinio()
    key = write_initial_global_weights(c, "bkt", seed=42)
    assert key == "round_0/global.pt"
    assert c.puts == ["round_0/global.pt"]


def test_written_weights_load_into_the_real_model():
    c = FakeMinio()
    write_initial_global_weights(c, "bkt", seed=42)
    model = ActorCritic()
    model.load_state_dict(_load(c, "round_0/global.pt"))  # raises if shapes mismatch


def test_same_seed_produces_identical_weights():
    a, b = FakeMinio(), FakeMinio()
    write_initial_global_weights(a, "bkt", seed=42)
    write_initial_global_weights(b, "bkt", seed=42)
    wa, wb = _load(a, "round_0/global.pt"), _load(b, "round_0/global.pt")
    for k in wa:
        assert torch.equal(wa[k], wb[k]), k


def test_different_seeds_produce_different_weights():
    a, b = FakeMinio(), FakeMinio()
    write_initial_global_weights(a, "bkt", seed=1)
    write_initial_global_weights(b, "bkt", seed=2)
    wa, wb = _load(a, "round_0/global.pt"), _load(b, "round_0/global.pt")
    assert any(not torch.equal(wa[k], wb[k]) for k in wa)


def test_is_idempotent_and_does_not_overwrite_existing_weights():
    # A retried pipeline head must not reset training back to round 0.
    c = FakeMinio(existing={"round_0/global.pt": b"sentinel"})
    key = write_initial_global_weights(c, "bkt", seed=42)
    assert key == "round_0/global.pt"
    assert c.puts == []
    assert c.objects["round_0/global.pt"] == b"sentinel"
