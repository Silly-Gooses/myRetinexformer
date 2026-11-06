"""Official Model D v1: frozen Model B plus HVI color correction."""

from .model import HVIColorHead, ModelD, load_model_d

__all__ = ["HVIColorHead", "ModelD", "load_model_d"]
