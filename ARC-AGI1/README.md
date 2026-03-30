Code to run our Qwen 4B model on ARC-AGI 2024 evaluation data.

The code is the same as in the notebook we used for our winning solution in Kaggle: [sorokin/arc2-qwen3-unsloth-flash-lora-batch4-queue](https://www.kaggle.com/code/sorokin/arc2-qwen3-unsloth-flash-lora-batch4-queue).

The variant in this folder moved all the installation scripts into the `pip-install-unsloth-flash-patch.ipynb` notebook. It should be run in Kaggle docker image available in December 2025.

The evaluation code is in the notebook: `002_ivan_arc1.ipynb` Theonly modifications compared to the kaggle notebook is to do test time fine tuning with the public evaluation data from arc prize 2024.



## Feature-flagged pseudo-test reranking (AR branch)

`arc_decoder.py` now supports optional pseudo-test leave-one-out reranking over grouped candidate outputs.

- Default behavior is unchanged.
- Enable with environment flag: `NVARC_ENABLE_PSEUDO_LOO_RERANK=1`.
- The reranker combines:
  1. existing base selection score,
  2. cheap structural checks (grid shape/palette sanity),
  3. pseudo-test leave-one-out consistency against train-pair profiles.

Use `ArcDecoder.run_selection_algo_with_metadata(...)` to get:
- `ranked_candidates` (attempt_1 should use index 0),
- `confidence` score,
- `candidate_debug` entries with `diversity_key` hook for future attempt_2 diversity routing.
