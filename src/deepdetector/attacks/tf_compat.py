"""TensorFlow 1 compatibility helpers for legacy attack code."""

from __future__ import annotations


def patch_tensorflow_v1_symbols() -> None:
    """Expose TF1 symbols expected by original Carlini attack implementations."""
    try:
        import tensorflow as tf
    except Exception:
        return

    try:
        tf.compat.v1.disable_eager_execution()
    except Exception:
        pass

    names = [
        "GraphKeys",
        "get_collection",
        "add_to_collection",
        "variable_scope",
        "get_variable",
        "global_variables",
        "local_variables",
        "trainable_variables",
        "global_variables_initializer",
        "variables_initializer",
        "placeholder",
        "Session",
        "ConfigProto",
        "reset_default_graph",
        "get_default_graph",
        "name_scope",
        "control_dependencies",
        "assign",
        "assign_add",
        "gradients",
    ]

    for name in names:
        if not hasattr(tf, name) and hasattr(tf.compat.v1, name):
            setattr(tf, name, getattr(tf.compat.v1, name))

    try:
        if not hasattr(tf.train, "AdamOptimizer"):
            tf.train.AdamOptimizer = tf.compat.v1.train.AdamOptimizer
    except Exception:
        pass
