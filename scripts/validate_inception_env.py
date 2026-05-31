"""
Validacao do ambiente adversarialimage-inceptionv3-tf2.

Execucao:
    conda activate adversarialimage-inceptionv3-tf2
    python scripts/validate_inception_env.py
"""
import sys
import numpy as np
import tensorflow as tf

tf.compat.v1.disable_eager_execution()

print(f"Python : {sys.version}")
print(f"TF     : {tf.__version__}")
print(f"GPUs   : {tf.config.list_physical_devices('GPU')}")

# Teste 1 — sess.run basico
g = tf.Graph()
with g.as_default():
    x = tf.compat.v1.placeholder(tf.float32, shape=(None, 4))
    y = x * 2.0
sess = tf.compat.v1.Session(graph=g)
out = sess.run(y, feed_dict={x: np.ones((2, 4), dtype=np.float32)})
assert out.shape == (2, 4)
np.testing.assert_allclose(out, 2.0, rtol=1e-5)
print("Teste 1 - sess.run basico: OK")

# Teste 2 — import_graph_def
ig = tf.Graph()
with ig.as_default():
    a = tf.compat.v1.placeholder(tf.float32, shape=(None, 299, 299, 3), name="inp")
    _ = a + 0.5
gdef = ig.as_graph_def()

og = tf.Graph()
with og.as_default():
    p = tf.compat.v1.placeholder(tf.float32, shape=(None, 299, 299, 3))
    elems = tf.compat.v1.graph_util.import_graph_def(
        gdef, name="im", input_map={"inp:0": p}, return_elements=["add:0"]
    )
sess2 = tf.compat.v1.Session(graph=og)
out2 = sess2.run(elems[0], feed_dict={p: np.zeros((1, 299, 299, 3), dtype=np.float32)})
assert out2.shape == (1, 299, 299, 3)
np.testing.assert_allclose(out2, 0.5, rtol=1e-5)
print("Teste 2 - import_graph_def + sess.run: OK")

# Teste 3 — cleverhans importavel
import cleverhans  # noqa: F401
print("Teste 3 - cleverhans importavel: OK")

print("\nAmbiente OK.")
