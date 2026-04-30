from __future__ import print_function

import importlib
import os
import sys


MODULES = [
    "caffe",
    "cleverhans",
    "keras",
    "matplotlib",
    "numpy",
    "pandas",
    "PIL",
    "scipy",
    "skimage",
    "tensorflow",
]


def main():
    print("Python:", sys.version.replace("\n", " "))
    print("PYTHONPATH:", os.environ.get("PYTHONPATH", ""))
    print("")

    failures = []
    for module_name in MODULES:
        try:
            module = importlib.import_module(module_name)
            version = getattr(module, "__version__", "unknown")
            print("[OK] {0} {1}".format(module_name, version))
        except Exception as exc:
            failures.append((module_name, exc))
            print("[FAIL] {0}: {1}".format(module_name, exc))

    if failures:
        print("")
        print("Missing or broken modules:")
        for module_name, exc in failures:
            print("- {0}: {1}".format(module_name, exc))
        return 1

    print("")
    print("Legacy environment looks ready.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
