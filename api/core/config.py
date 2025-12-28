# api/core/config.py
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="FPN_", env_file=".env", extra="ignore")

    # Paths
    repo_root: str = "."
    artifacts_dir: str = "artifacts"
    data_dir: str = "data/processed"
    logs_dir: str = "logs"

    # -------------------------
    # Artifact paths (NEW layout)
    # -------------------------
    # models/
    xgb_model_path: str = "models/xgb_model.pkl"
    xgb_model_feedback_path: str = "models/xgb_model_feedback.pkl"
    ae_model_path: str = "models/autoencoder_model.keras"

    # preprocess/
    preprocess_path: str = "preprocess/preprocess.joblib"
    features_path: str = "preprocess/features.json"

    # thresholds/
    ae_thresholds_path: str = "thresholds/ae_thresholds.json"
    ae_threshold_txt_path: str = "thresholds/ae_threshold.txt"

    # ae_errors/
    ae_baseline_path: str = "ae_errors/ae_baseline_legit_errors.npy"
    ae_val_errors_path: str = "ae_errors/ae_val_errors.npy"
    ae_test_errors_path: str = "ae_errors/ae_test_errors.npy"

    # snapshots/
    feature_snapshots_dir: str = "snapshots/feature_snapshots"

    # stores/
    sqlite_path: str = "artifacts/stores/inference_store.sqlite"
    feedback_log_path: str = "artifacts/stores/feedback_log.jsonl"
    review_queue_jsonl_path: str = "artifacts/stores/review_queue.jsonl"

    # Logging
    log_path: str = "logs/inference.jsonl"

    # Thresholds (decision logic)
    xgb_t_low: float = 0.05
    xgb_t_high: float = 0.80

    # Runtime toggles
    strict_feature_check: bool = True

    # --- derived helpers ---
    def root_path(self) -> Path:
        return Path(self.repo_root).resolve()

    def artifacts_path(self) -> Path:
        return (self.root_path() / self.artifacts_dir).resolve()

    def logs_path(self) -> Path:
        return (self.root_path() / self.logs_dir).resolve()

    def abs_log_path(self) -> Path:
        return (self.root_path() / self.log_path).resolve()

    def abs_sqlite_path(self) -> Path:
        return (self.root_path() / self.sqlite_path).resolve()

    def abs_feedback_log_path(self) -> Path:
        return (self.root_path() / self.feedback_log_path).resolve()

    def abs_review_queue_jsonl_path(self) -> Path:
        return (self.root_path() / self.review_queue_jsonl_path).resolve()

    def abs_feature_snapshots_dir(self) -> Path:
        # -> <repo>/artifacts/snapshots/feature_snapshots
        return (self.artifacts_path() / self.feature_snapshots_dir).resolve()

    # convenience: absolute artifact file paths
    def abs_xgb_model_path(self) -> Path:
        return (self.artifacts_path() / self.xgb_model_path).resolve()

    def abs_xgb_model_feedback_path(self) -> Path:
        return (self.artifacts_path() / self.xgb_model_feedback_path).resolve()

    def abs_ae_model_path(self) -> Path:
        return (self.artifacts_path() / self.ae_model_path).resolve()

    def abs_preprocess_path(self) -> Path:
        return (self.artifacts_path() / self.preprocess_path).resolve()

    def abs_features_path(self) -> Path:
        return (self.artifacts_path() / self.features_path).resolve()

    def abs_ae_thresholds_path(self) -> Path:
        return (self.artifacts_path() / self.ae_thresholds_path).resolve()

    def abs_ae_baseline_path(self) -> Path:
        return (self.artifacts_path() / self.ae_baseline_path).resolve()


@lru_cache
def get_settings() -> Settings:
    return Settings()
