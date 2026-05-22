# MNIST M2 Clean Baseline

## Configuration

- dataset: MNIST
- model: M2 CNN base architecture
- epochs: 10
- batch_size: 128
- learning_rate: 0.001
- train_dir: `C:\Users\Eduar\OneDrive\Documentos\GitHub\AdversarialImage-IDS\results\mnist\m2_cw\clean_baseline\checkpoints`
- filename: `mnist_m2.ckpt`
- trained_from_scratch: False
- seed_tf: 1234
- seed_numpy: 20170830

## Metrics

- test_accuracy_clean: 0.993800

## Checkpoint

`C:\Users\Eduar\OneDrive\Documentos\GitHub\AdversarialImage-IDS\results\mnist\m2_cw\clean_baseline\checkpoints\mnist_m2.ckpt`

## Notes

The exact M2 architecture from reference [36] is not present in this repository. This checkpoint uses the base M2 CNN definition aligned in `src/deepdetector/models/mnist_m2.py`.
