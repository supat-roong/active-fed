from src.pipelines.run_pipeline import compute_start_round


class FakeObj:
    def __init__(self, name):
        self.object_name = name


class FakeMinio:
    def __init__(self, names):
        self._names = names

    def list_objects(self, bucket, prefix="", recursive=False):
        return [FakeObj(n) for n in self._names if n.startswith(prefix)]


def test_empty_bucket_starts_at_zero():
    assert compute_start_round(FakeMinio([]), "b") == 0


def test_only_initial_weights_starts_at_zero():
    assert compute_start_round(FakeMinio(["round_0/global.pt"]), "b") == 0


def test_resumes_at_highest_completed_round():
    names = ["round_0/global.pt", "round_1/global.pt", "round_2/global.pt"]
    assert compute_start_round(FakeMinio(names), "b") == 2


def test_ignores_worker_artifacts():
    names = [
        "round_0/global.pt",
        "round_1/global.pt",
        "round_1/workers/worker_0_weights.pt",
        "round_1/workers/worker_1_delta.pt",
    ]
    assert compute_start_round(FakeMinio(names), "b") == 1


def test_ignores_unparseable_keys():
    names = ["round_0/global.pt", "roundX/global.pt", "scratch/global.pt", "round_3/global.pt"]
    assert compute_start_round(FakeMinio(names), "b") == 3


def test_non_contiguous_rounds_resume_at_the_maximum():
    # Gaps mean an earlier attempt died mid-experiment; the newest checkpoint wins.
    assert compute_start_round(FakeMinio(["round_0/global.pt", "round_5/global.pt"]), "b") == 5
