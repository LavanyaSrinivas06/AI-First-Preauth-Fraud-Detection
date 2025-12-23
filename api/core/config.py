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

    # Model artifact filenames
    xgb_model_path: str = "xgb_model.pkl"
    preprocess_path: str = "preprocess.joblib"
    features_path: str = "features.json"
    ae_model_path: str = "autoencoder_model.keras"
    ae_thresholds_path: str = "ae_thresholds.json"

    # Decision params
    alpha: float = 0.7               # weight if you later fuse scores smoothly
    xgb_t_low: float = 0.25
    xgb_t_high: float = 0.75

    # Storage
    sqlite_path: str = "artifacts/inference_store.sqlite"

    # Logging
    log_path: str = "logs/inference.jsonl"

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


@lru_cache
def get_settings() -> Settings:
    return Settings()
