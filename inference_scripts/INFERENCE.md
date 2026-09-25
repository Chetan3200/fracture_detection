# Inference

Use root `infer.py` as documented in the [repository README](../README.md#6-predict-images-or-folders-without-ground-truth).

Inference shares the byte-stable checkpoint and MedGemma helpers from `evaluation_scripts/`. Protocol mode checks the matching checkpoint, settings, software, and helper hashes. It does not load ground truth or compute evaluation metrics.
