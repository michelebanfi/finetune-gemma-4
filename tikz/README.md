---
license: gemma
language:
  - en
library_name: gguf
tags:
  - gemma
  - gemma-4
  - tikz
  - latex
  - qlora
  - unsloth
  - gguf
  - text-generation
base_model: unsloth/gemma-4-E4B-it
datasets:
  - nllg/DaTikZ-V4
pipeline_tag: text-generation
---

# gemma4-4b-tikz

> [!NOTE]
> Draft model card placeholder for the TikZ fine-tune. Update hub links once the HuggingFace repo is created.

**gemma4-4b-tikz** is a TikZ-focused fine-tune of Gemma 4, trained to improve generation of compilable TikZ/LaTeX figures from scientific captions and multimodal inputs.

## Model Description

- **Developed by:** Michele Banfi
- **Base model:** `unsloth/gemma-4-E4B-it`
- **Method:** QLoRA (4-bit) + SFT via Unsloth
- **Dataset:** `nllg/DaTikZ-V4`
- **License:** [Gemma Terms of Use](https://ai.google.dev/gemma/terms)

## Intended Use

- Generate TikZ/LaTeX code from scientific figure descriptions
- Improve robustness of TikZ syntax for compilable outputs
- Support multimodal TikZ generation workflows

## Training Setup (Current)

- 1 epoch (configurable in `tikz/config.yaml`)
- Mixed task training:
  - caption_to_code
  - image_to_description
  - image_to_code

## Evaluation

Primary metric is compilation success rate on `nllg/datikz-v3` test split, with optional visual similarity metrics (SSIM/LPIPS) for successful compilations.

## Model Sources

- **Repository:** https://github.com/michelebanfi/finetune-gemma-4
- **Sub-project docs:** `tikz/`

## Citation

```bibtex
@misc{banfi2026gemma4tikz,
  title={Gemma-4 TikZ Fine-tuning},
  author={Banfi, Michele},
  year={2026},
  howpublished={\url{https://github.com/michelebanfi/finetune-gemma-4}}
}
```
