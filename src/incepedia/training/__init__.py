"""nanotron-based training entry points for 1.82B Llama2 ablation runs.

Public API
----------
    ExperimentConfig, load_config          — pydantic schema, YAML loader
    Track, Stage                           — enums
    launch_training                        — render + launch nanotron training

Lazy imports: nanotron is imported only inside `launch_training`, so you can
use this package for config validation / orchestration without flash-attn.
"""

from incepedia.training.config import (
    ExperimentConfig,
    Stage,
    Track,
    load_config,
)

__all__ = ["ExperimentConfig", "Stage", "Track", "load_config", "launch_training"]


def launch_training(*args, **kwargs):
    """Proxy to avoid eager import of nanotron from `launcher`."""
    from incepedia.training.launcher import launch_training as _impl
    return _impl(*args, **kwargs)
