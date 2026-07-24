from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SQLITE_PATH = (
    PROJECT_ROOT / "01_data_collection_cleaning" / "interim" / "ercot_data.sqlite"
)
DEFAULT_REALTIME_SQLITE_PATH = PROJECT_ROOT / "05_realtime_service" / "data" / "realtime.sqlite"
DEFAULT_MODEL_DIR = (
    PROJECT_ROOT
    / "02_model_training_validation"
    / "02_model_training_validation"
    / "models"
    / "c1_prediction_agent"
)


def get_database_uri() -> str:
    return os.getenv("ERCOT_AGENT_DB_URI", f"sqlite:///{DEFAULT_SQLITE_PATH.as_posix()}")


def get_realtime_database_uri() -> str:
    return os.getenv(
        "ERCOT_REALTIME_DB_URI",
        f"sqlite:///{DEFAULT_REALTIME_SQLITE_PATH.as_posix()}",
    )


def get_agent_db_uri() -> str:
    return get_database_uri()


def get_model_dir() -> Path:
    return Path(os.getenv("ERCOT_MODEL_DIR", str(DEFAULT_MODEL_DIR)))


def get_model_fold() -> str:
    return os.getenv("ERCOT_MODEL_FOLD", "validation_fold_3")
