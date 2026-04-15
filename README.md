---
license: gemma
language:
  - en
library_name: gguf
tags:
  - gemma
  - gemma-4
  - scientific
  - science
  - research
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
> **Work in Progress.** This model is an early-stage research experiment trained for only 600 steps on a small subset of the available data. Draft ScholarQABench Tier 1 results are available (see Evaluation section) but it has not been fully evaluated and should not be relied upon for any critical purpose. Expect rough edges, hallucinations, and factual errors. The roadmap section below describes planned improvements.

## Model Summary

**gemma4-4b-sci** is a scientific-domain fine-tune of Google's [Gemma 4 E4B instruction-tuned model](https://huggingface.co/unsloth/gemma-4-E4B-it), trained via QLoRA (4-bit) + supervised fine-tuning (SFT) on 30,000 scientific instruction examples drawn from the [OpenSciLM training corpus](https://huggingface.co/datasets/OpenSciLM/OS_Train_Data) and [Allen AI's SciRIFF](https://huggingface.co/datasets/allenai/SciRIFF-train-mix). The goal is a lightweight, Ollama-ready model capable of answering research-level scientific questions across domains such as biology, physics, climate science, and more. The GGUF export (Q8_0) can be run locally via Ollama on consumer hardware.

## Inspiration and Acknowledgements

This work is directly inspired by and builds upon:

- **[OpenScholar](https://allenai.org/blog/nature-openscilm)** — _OpenScholar: Synthesizing Scientific Literature with Retrieval-Augmented LMs_ by Asai et al. (2024, Allen Institute for AI). OpenScholar demonstrated that LLMs fine-tuned on curated scientific instruction data can synthesize research literature at expert level. The training corpus used here (`OpenSciLM/OS_Train_Data`) originates from that project.

- **[`OpenSciLM/Llama-3.1_OpenScholar-8B`](https://huggingface.co/OpenSciLM/Llama-3.1_OpenScholar-8B)** — the reference OpenScholar model fine-tuned on Llama 3.1 8B, which serves as the methodological blueprint for this effort.

> **Important distinction from OpenScholar:** The original OpenScholar system is a full retrieval-augmented generation (RAG) pipeline that grounds responses in a live corpus of 45M+ papers. This model is **not** a RAG system — it is a parametric fine-tune only. It will not cite real papers reliably and cannot retrieve up-to-date research. See [Limitations](#limitations) below.

## Intended Uses

**Suitable for:**

- Research assistance and scientific Q&A (qualitative exploration, not authoritative answers)
- Summarizing and explaining scientific concepts
- Draft generation for scientific writing (with human review)
- Experimentation and research into scientific LLM fine-tuning

**Out of scope / not recommended:**

- Clinical, medical, or legal decision-making
- Any application requiring verifiable citations (the model can hallucinate references)
- Production deployment without further evaluation and alignment work
- Replacing domain experts or peer review

## How to Use

### Ollama (recommended)

```bash
ollama run hf.co/linosium/gemma4-4b-sci
```

### Transformers (fp16 merged weights)

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_id = "linosium/gemma4-4b-sci"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

messages = [
    {"role": "user", "content": "Explain the role of CRISPR-Cas9 in gene editing and its current limitations."}
]

input_ids = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt"
).to(model.device)

output = model.generate(input_ids, max_new_tokens=512, temperature=1.0, top_p=0.95, top_k=64)
print(tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True))
```

## Training Details

### Base Model

|                   |                                              |
| ----------------- | -------------------------------------------- |
| Model             | `unsloth/gemma-4-E4B-it`                     |
| Architecture      | Gemma 4 (E4B instruction-tuned), multimodal  |
| Fine-tuned layers | Language layers only (vision encoder frozen) |

### Method

- **QLoRA** (4-bit quantized base + float16 adapter) via [Unsloth](https://github.com/unslothai/unsloth) `FastModel`
- **Supervised Fine-Tuning (SFT)** using HuggingFace TRL's `SFTTrainer`
- **Response-only training**: loss is computed only on the model's responses; user turns are masked

### LoRA Configuration

| Parameter      | Value                                  |
| -------------- | -------------------------------------- |
| `lora_r`       | 16                                     |
| `lora_alpha`   | 32                                     |
| `lora_dropout` | 0                                      |
| `bias`         | none                                   |
| Target modules | Attention + MLP (language layers only) |
| Vision layers  | Frozen (not fine-tuned)                |

### Training Hyperparameters

| Parameter                | Value                       |
| ------------------------ | --------------------------- |
| Learning rate            | 2e-4                        |
| LR scheduler             | Cosine                      |
| Warmup steps             | 60                          |
| Optimizer                | `adamw_8bit`                |
| Weight decay             | 0.01                        |
| Max sequence length      | 4096                        |
| Per-device batch size    | 1                           |
| Gradient accumulation    | 16                          |
| **Effective batch size** | **16**                      |
| Number of epochs         | 1 (capped at **600 steps**) |
| Precision                | bf16                        |
| Seed                     | 42                          |

### Hardware

Trained on a single **NVIDIA RTX 5090 (32 GB VRAM)**.

## Training Data

| Dataset                                                                                  | Samples used | Total available |
| ---------------------------------------------------------------------------------------- | ------------ | --------------- |
| [`OpenSciLM/OS_Train_Data`](https://huggingface.co/datasets/OpenSciLM/OS_Train_Data)     | 15,000       | ~130,000        |
| [`allenai/SciRIFF-train-mix`](https://huggingface.co/datasets/allenai/SciRIFF-train-mix) | 15,000       | ~70,000         |
| **Total**                                                                                | **30,000**   | —               |

Both datasets were shuffled (seed=42) and normalized into Gemma 4's native chat format (role `assistant` remapped to `model`).

- **OS_Train_Data** contains scientific instruction-following examples curated for the OpenScholar project, covering reading comprehension, summarization, and Q&A over scientific literature.
- **SciRIFF** (Scientific Relation and Information Formulation Format) from Allen AI covers a broad set of scientific NLP tasks across multiple domains and paper corpora.

## Evaluation

### ScholarQABench (Tier 1 — single-paper tasks)

> [!NOTE]
> **Draft results** from a 600-step training run on 30K examples. The model has no retrieval pipeline, so only Tier 1 tasks (where gold paper contexts are provided) are evaluated here. Tier 2 synthesis tasks (ScholarQA-CS/Bio/Neuro) require a live document corpus and are left for a future RAG-augmented variant. Citation scores are near zero because the model was not explicitly trained to cite retrieved passages — this is expected and is the primary gap vs. OpenScholar.

Evaluated using [ScholarQABench](https://github.com/AkariAsai/ScholarQABench) with `temperature=0.1`, `max_new_tokens=256/512`, batch size 8 on an NVIDIA RTX 5090.

#### Tier 1 — single-paper tasks (gold context provided, fair comparison)

| Task           | Metric      | gemma4-4b-sci | OpenScholar-8B | Delta |
| -------------- | ----------- | ------------: | -------------: | ----: |
| SciFact (208)  | Accuracy    |     **77.9%** |          76.4% |  +1.5 |
| PubMedQA (843) | Accuracy    |     **81.5%** |          76.0% |  +5.5 |
| QASA (1375)    | ROUGE-L     |          20.9 |           23.0 |  −2.1 |
| SciFact        | Citation F1 |           0.0 |           68.9 | −68.9 |
| PubMedQA       | Citation F1 |           0.0 |           43.6 | −43.6 |
| QASA           | Citation F1 |           4.3 |           56.3 | −52.0 |

**Key takeaway:** Correctness scores are competitive with or exceed OpenScholar-8B (an 8B model) despite this being a 4B model trained for only 600 steps. The citation gap is entirely explained by the absence of a retrieval pipeline.

#### Tier 2 — multi-document synthesis (no retrieval, informational only)

> These tasks require synthesizing across many papers. Without a retrieval pipeline, the model answers from parametric knowledge only. Citation recall/precision of 0.0 is expected and matches the no-retrieval Llama-8B baseline in the OpenScholar paper. The model generates ~17–19 citation brackets per answer (it learned the citation style from training data) but they cannot be verified without real retrieved documents.

| Task                   | Metric           | gemma4-4b-sci | OpenScholar-8B |
| ---------------------- | ---------------- | ------------: | -------------: |
| ScholarQA-CS (110)     | Citation F1      |           0.0 |           47.9 |
| ScholarQA-CS (110)     | Avg. citations   |          18.7 |            —   |
| ScholarQA-Bio (1451)   | Citation F1      |           0.0 |           42.8 |
| ScholarQA-Bio (1451)   | Avg. citations   |          19.3 |            —   |
| ScholarQA-Neuro (1308) | Citation F1      |           0.0 |           50.8 |
| ScholarQA-Neuro (1308) | Avg. citations   |          17.0 |            —   |

Rubric correctness scores for ScholarQA-CS (which require a GPT-4 judge) are not reported here. Adding a retrieval frontend is the necessary next step to make Tier 2 citation scores meaningful.

### Qualitative comparison

A side-by-side comparison between the base Gemma-4 model and the fine-tuned model on 5 open-ended scientific questions is run automatically after training and saved to `{output_dir}/test_results_*.txt`. Questions cover:

1. CRISPR-Cas9 mechanisms and limitations
2. AlphaFold2's approach to protein structure prediction
3. Statistical analysis in clinical trials (p-values, confidence intervals, effect size)
4. Dark matter candidates in particle physics
5. Positive vs. negative climate feedback loops

Generation parameters: `temperature=1.0`, `top_p=0.95`, `top_k=64`, `max_new_tokens=512`.

## Model Formats

| Format                     | Notes                                      |
| -------------------------- | ------------------------------------------ |
| LoRA adapter (safetensors) | Applies on top of `unsloth/gemma-4-E4B-it` |
| Merged fp16 (safetensors)  | Full model weights                         |
| **Q8_0 GGUF**              | **Primary release format; Ollama-ready**   |

> Gemma 4's GGUF export is currently restricted to Q8_0, BF16, and F16 quantizations via `llama.cpp`. Lower-bit quantizations (Q4_K_M, Q5_K_M, etc.) will be added once support is available upstream.

## Limitations

- **Short training run**: 600 steps on 30K examples is a minimal proof-of-concept. The model is likely undertrained.
- **No RAG grounding**: Unlike the original OpenScholar, this model has no access to a live paper corpus and cannot reliably cite specific papers. Treat any citation it produces with skepticism.
- **Hallucination risk**: The model may confidently produce plausible-sounding but incorrect scientific statements.
- **English only**: Training data and testing are English-only.
- **Knowledge cutoff**: Inherits the knowledge cutoff of the base Gemma 4 model; not updated with recent literature.
- **No RLHF / alignment**: No preference optimization has been applied beyond the base model's instruction tuning.
- **4B parameter scale**: Significantly smaller than state-of-the-art frontier models; expect weaker reasoning on complex multi-step problems.

## Roadmap

- [ ] Incorporate all available `OS_Train_Data` and `SciRIFF` examples (not just 15K subsets)
- [x] Run **ScholarQABench** Tier 1 evaluation and publish results
- [ ] Run **ScholarQABench** Tier 2 synthesis tasks (requires retrieval frontend)
- [ ] Explore retrieval-augmented grounding (RAG pipeline)
- [ ] Additional GGUF quantization levels once `llama.cpp` supports them for Gemma 4
- [ ] DPO or preference optimization pass

## Citation

If you use this model, please also cite the underlying works that made it possible:

```bibtex
@article{asai2024openscholar,
  title     = {OpenScholar: Synthesizing Scientific Literature with Retrieval-Augmented LMs},
  author    = {Asai, Akari and He, Jacqueline and Shao, Rulin and Shi, Weijia and Singh, Amanpreet and Chang, Joseph Chee and Lo, Kyle and Soldaini, Luca and Feldman, Sergey and D'Arcy, Mike and Wadden, David and Latzke, Matt and Minyang Jiang and Ji, Pan and Liu, Shengding and Shi, Hao and Gu, Wanjun and Murray, John and Chen, Yuze and Subramani, Nishant and Zettlemoyer, Luke and Neubig, Graham and Weld, Daniel and Downey, Doug and Ha, Daniel and Hajishirzi, Hannaneh and Koh, Pang Wei},
  journal   = {Nature},
  year      = {2024},
  url       = {https://allenai.org/blog/nature-openscilm}
}

@article{wadden2024sciriff,
  title     = {SciRIFF: A Resource to Enhance Language Model Instruction-Following over Scientific Literature},
  author    = {Wadden, David and Pan, Kejian and Shi, Hao and Ajith, Aakanksha and Latzke, Matt and Soldaini, Luca and Lo, Kyle and Weld, Daniel and Hope, Tom and Hajishirzi, Hannaneh},
  year      = {2024},
  url       = {https://huggingface.co/datasets/allenai/SciRIFF-train-mix}
}

@article{gemmateam2024gemma,
  title     = {Gemma: Open Models Based on Gemini Research and Technology},
  author    = {{Gemma Team}},
  year      = {2024},
  url       = {https://ai.google.dev/gemma}
}

@software{unsloth2024,
  title     = {Unsloth},
  author    = {Han, Daniel and Han, Michael},
  year      = {2024},
  url       = {https://github.com/unslothai/unsloth}
}
```

## License

This model is released under the [Gemma Terms of Use](https://ai.google.dev/gemma/terms). Use is subject to Google's Gemma license. The training datasets retain their respective licenses:

- `OpenSciLM/OS_Train_Data`: see [OpenSciLM dataset page](https://huggingface.co/datasets/OpenSciLM/OS_Train_Data)
- `allenai/SciRIFF-train-mix`: ODC-BY (Open Data Commons Attribution License)
