import importlib

MODULES = [
    "src.config",
    "src.data",
    "src.preprocessing",
    "src.clustering",
    "src.zones",
    "src.visualization",
]


def test_src_modules_import():
    for name in MODULES:
        assert importlib.import_module(name) is not None
