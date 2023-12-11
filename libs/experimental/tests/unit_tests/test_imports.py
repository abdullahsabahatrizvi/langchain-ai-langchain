import glob
import importlib


def test_importable_all() -> None:
    for path in glob.glob("../experimental/langchain_experimental/*"):
        relative_path = path.split("/")[-1]
        if relative_path.endswith(".typed"):
            continue
        module_name = relative_path.split(".")[0]
        module = importlib.import_module("langchain_experimental." + module_name)
        all_ = getattr(module, "__all__", [])
        for cls_ in all_:
            getattr(module, cls_)
