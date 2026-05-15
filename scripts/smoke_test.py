from __future__ import print_function

from importlib import import_module

try:
    from importlib.metadata import PackageNotFoundError, version
except ImportError:
    from pkg_resources import DistributionNotFound as PackageNotFoundError
    from pkg_resources import get_distribution

    def version(package_name):
        return get_distribution(package_name).version


REQUIRED_IMPORTS = (
    "tensorflow",
    "keras",
    "cleverhans",
    "numpy",
)


def get_version(module_name, module):
    """Return the installed package version when available."""
    try:
        return version(module_name)
    except PackageNotFoundError:
        return getattr(module, "__version__", "unknown")


def main():
    """Import the legacy runtime dependencies and print their versions."""
    for module_name in REQUIRED_IMPORTS:
        module = import_module(module_name)
        module_version = get_version(module_name, module)
        print("{0}: {1}".format(module_name, module_version))


if __name__ == "__main__":
    main()
