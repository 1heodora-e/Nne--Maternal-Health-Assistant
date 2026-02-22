# Dataset

**File:** `nne_mixed_train_v2.jsonl`

Preprocessed maternal-health instruction dataset used to fine-tune Nne (Gemma-2-2b-it with LoRA).

- **Sources:** Medical Meadow, Mental Health Counseling, ChatDoctor (maternal-health subset).
- **Size:** 11,674 instruction–input–output triples.
- **Splits:** Produced in the notebook (train/val/test, e.g. 80/10/10).
- **Fields per line:** `instruction`, `input`, `output` (output may include `<thinking>Risk: ...</thinking>`).

See the main [README](../README.md) and `nne-llm-fine-tuning-pipeline.ipynb` for how the dataset is built and used.
