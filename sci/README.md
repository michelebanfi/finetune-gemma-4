---
license: gemma
language:
  - en
library_name: gguf
tags:
  - gemma
  - gemma-4
  - scientific
  - qlora
  - unsloth
  - gguf
  - ollama
  - openscholar
  - sciriff
  - text-generation
base_model: unsloth/gemma-4-E4B-it
datasets:
  - OpenSciLM/OS_Train_Data
  - allenai/SciRIFF-train-mix
pipeline_tag: text-generation
---

# gemma4-4b-sci

> [!WARNING]
> Early-stage research experiment. Trained for 1 epoch on 30K examples. Expect hallucinations and factual errors.

**gemma4-4b-sci** is a scientific-domain fine-tune of [Gemma 4 E4B](https://huggingface.co/unsloth/gemma-4-E4B-it) via QLoRA on 30,000 examples from [OpenSciLM/OS_Train_Data](https://huggingface.co/datasets/OpenSciLM/OS_Train_Data) and [SciRIFF](https://huggingface.co/datasets/allenai/SciRIFF-train-mix). Inspired by [OpenScholar](https://allenai.org/blog/nature-openscilm) — this is a **generation-only** model without a retrieval pipeline.

### Model Description

- **Developed by:** Michele Banfi
- **Base model:** `unsloth/gemma-4-E4B-it`
- **Method:** QLoRA (4-bit) + SFT via Unsloth, language layers only (vision encoder frozen)
- **Training:** 1 epoch, 30K examples (15K OS_Train_Data + 15K SciRIFF), NVIDIA RTX 5090
- **License:** [Gemma Terms of Use](https://ai.google.dev/gemma/terms)

### Model Sources

- **Repository:** https://github.com/michelebanfi/gemma-4-finetuning
- **Evaluation:** [ScholarQABench](https://github.com/AkariAsai/ScholarQABench)
- **Ollama:** `ollama run hf.co/michelinolinolino/gemma4-4b-sci`

## Quick Start

```bash
ollama run hf.co/michelinolinolino/gemma4-4b-sci
```

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model = AutoModelForCausalLM.from_pretrained("michelinolinolino/gemma4-4b-sci", torch_dtype=torch.bfloat16, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("michelinolinolino/gemma4-4b-sci")

messages = [{"role": "user", "content": "Explain the role of CRISPR-Cas9 in gene editing."}]
input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)
print(tokenizer.decode(model.generate(input_ids, max_new_tokens=512)[0][input_ids.shape[1]:], skip_special_tokens=True))
```

## Evaluation

[ScholarQABench](https://github.com/AkariAsai/ScholarQABench) — draft results, 1-epoch run. Gold paper contexts provided (fair comparison with OpenScholar-8B).

| Task | Metric | gemma4-4b-sci | OpenScholar-8B |
|---|---|---:|---:|
| SciFact (208) | Accuracy | **77.9%** | 76.4% |
| PubMedQA (843) | Accuracy | **81.5%** | 76.0% |
| QASA (1375) | ROUGE-L | 20.9 | 23.0 |
| SciFact | Citation F1 | 0.0 | 68.9 |
| PubMedQA | Citation F1 | 0.0 | 43.6 |
| QASA | Citation F1 | 4.3 | 56.3 |

Correctness matches or exceeds OpenScholar-8B (2× the parameters) at 1 epoch. Citation gap is entirely due to the missing retrieval pipeline.

## Citation

```bibtex
@article{asai2024openscholar,
  title   = {OpenScholar: Synthesizing Scientific Literature with Retrieval-Augmented LMs},
  author  = {Asai, Akari and others},
  journal = {Nature},
  year    = {2024},
  url     = {https://allenai.org/blog/nature-openscilm}
}
```
