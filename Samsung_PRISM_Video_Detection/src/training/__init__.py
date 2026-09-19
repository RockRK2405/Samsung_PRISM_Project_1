"""Training + evaluation entry points.

The face-track trainer (``trainer.py``) depends on mlflow. The general
AI-video trainer (``video_trainer.py``) does not, so it is imported
directly by its scripts. Guard the mlflow-dependent import so this
package loads cleanly on a lean GPU box without mlflow installed.
"""

__all__ = []

try:
    from src.training.trainer import run_evaluation, run_training

    __all__ += ["run_evaluation", "run_training"]
except ImportError:
    # mlflow not installed — fine for the general AI-video track.
    pass
