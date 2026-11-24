import importlib

__version__ = "0.1.2"


def __getattr__(name):
    if name in ("connect", "start", "Run", "Client"):
        return getattr(importlib.import_module(".api", __name__), name)
    if name == "losses":
        return importlib.import_module(".losses", __name__)
    raise AttributeError(name)
