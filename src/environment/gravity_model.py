"""Read and validate frozen behavior inputs without importing research runners."""
import hashlib
import json
from pathlib import Path
from src import config as P


ROOT = Path(__file__).resolve().parents[2]
MODEL_VERSION = 2
TRAVEL_COST_BASIS = "baseline_user_equilibrium_congested_shortest_path_v1"
GRAVITY_UE_TARGET_RELATIVE_GAP = float(P.UE_RGAP_DEF)
GRAVITY_UE_MAX_ITERATIONS = 2000
DEFAULT_MODEL = ROOT / "data/siouxfalls_toy/behavior/gravity_model.json"
MODEL_SOURCES = (
    ROOT / "data/siouxfalls_toy/network/od_pairs.csv",
    ROOT / "data/siouxfalls_toy/network/edges.csv",
)


def _source_fingerprint() -> dict[str, str]:
    """Hashes of the raw inputs that define the fitted model."""
    return {
        str(path.relative_to(ROOT)).replace("\\", "/"): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in MODEL_SOURCES
    }

def _artifact_is_current(artifact: dict) -> bool:
    return (
        artifact.get("model_version") == MODEL_VERSION
        and artifact.get("travel_cost_basis") == TRAVEL_COST_BASIS
        and artifact.get("fit_settings")
        == {
            "ue_target_relative_gap": GRAVITY_UE_TARGET_RELATIVE_GAP,
            "ue_max_iterations": GRAVITY_UE_MAX_ITERATIONS,
        }
        and artifact.get("source_sha256") == _source_fingerprint()
        and set(artifact.get("coefficients", {}))
        == {"intercept", "log_origin_total", "log_destination_total", "travel_cost_minutes"}
    )

def load_gravity_model(model_path: Path = DEFAULT_MODEL) -> dict:
    """Load the shared fitted model and reject stale or incompatible artifacts."""
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(
            f"gravity model is missing at {model_path}; restore the approved frozen input, "
            "or explicitly provide a separately approved model path"
        )
    artifact = json.loads(model_path.read_text(encoding="utf-8"))
    if not _artifact_is_current(artifact):
        raise RuntimeError(
            f"gravity model at {model_path} does not match the current raw inputs or schema"
        )
    artifact["path"] = str(model_path)
    return artifact
