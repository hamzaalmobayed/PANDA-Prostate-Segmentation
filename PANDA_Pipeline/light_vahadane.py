"""TIAToolbox's VahadaneNormalizer without importing the rest of TIAToolbox.

`import tiatoolbox.tools.stainnorm` runs tiatoolbox/tools/__init__.py and
tiatoolbox/utils/__init__.py, which import every tool and pull in torch and
TensorFlow (about 0.9 GB per process). Normalization worker processes only
need the stain-normalization code, so this module registers the two package
folders without running their __init__ files and then imports the original
stainnorm module. The class used is TIAToolbox's own VahadaneNormalizer.
"""
import importlib
import importlib.util
import sys
import types


def _bare_package(name):
    if name in sys.modules:
        return
    spec = importlib.util.find_spec(name)
    mod = types.ModuleType(name)
    mod.__path__ = list(spec.submodule_search_locations)
    mod.__spec__ = spec
    sys.modules[name] = mod


def vahadane_normalizer_class():
    importlib.import_module("tiatoolbox")          # light: logger and config only
    _bare_package("tiatoolbox.tools")
    _bare_package("tiatoolbox.utils")
    return importlib.import_module("tiatoolbox.tools.stainnorm").VahadaneNormalizer


class VahadaneProcessor:
    """Same behaviour as panda_camelyon_style_pipeline.VahadaneProcessor."""

    def __init__(self, target_rgb):
        import numpy as np
        self.np = np
        self.norm = vahadane_normalizer_class()()
        self.norm.fit(np.array(target_rgb, dtype=np.uint8, copy=True, order="C"))

    def transform(self, rgb):
        np = self.np
        out = self.norm.transform(np.array(rgb, dtype=np.uint8, copy=True, order="C"))
        return np.array(out, dtype=np.uint8, copy=True)
