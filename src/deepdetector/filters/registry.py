"""Central registry of stateless image filters."""

from __future__ import print_function

from collections import OrderedDict
from functools import partial

from deepdetector.filters.adaptive_quantization import entropy_based_quantization
from deepdetector.filters.article_final import article_final_detection_filter
from deepdetector.filters.mean_filters import (
    box_mean_filter,
    cross_mean_filter,
    diamond_mean_filter,
)
from deepdetector.filters.quantization import nonuniform_quantization, scalar_quantization


FILTER_REGISTRY = OrderedDict(
    [
        ("scalar_128", partial(scalar_quantization, interval=128)),
        ("scalar_64", partial(scalar_quantization, interval=64)),
        ("scalar_43", partial(scalar_quantization, interval=43)),
        ("nonuniform", nonuniform_quantization),
        ("entropy_adaptive", lambda img: entropy_based_quantization(img)[0]),
        ("box_3", partial(box_mean_filter, kernel_size=3)),
        ("box_5", partial(box_mean_filter, kernel_size=5)),
        ("cross_3", partial(cross_mean_filter, radius=1)),
        ("diamond_3", partial(diamond_mean_filter, radius=1)),
        ("article_final", article_final_detection_filter),
    ]
)
