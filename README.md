# **Translation of Text Embedding via Delta Vector to Suppress Strongly-Entangled Content in Text-to-Image Diffusion Models**

[![arXiv](https://img.shields.io/badge/arXiv-SSDV-<COLOR>.svg)](https://arxiv.org/abs/2508.10407) [![arXiv](https://img.shields.io/badge/paper-SSDV-b31b1b.svg)](https://arxiv.org/pdf/2508.10407.pdf)

This is the official implementation for the paper “[ICCV 2025] Translation of Text Embedding via Delta Vector to Suppress Strongly-Entangled Content in Text-to-Image Diffusion Models”.


<div align="left">
    <img width="100%" alt="teaser" src="https://github.com/eunso999/SSDV/blob/main/source/examples.png?raw=true"/>
</div>

## Abstract

Text-to-Image (T2I) diffusion models have made significant progress in generating diverse high-quality images from textual prompts. However, these models still face challenges in suppressing content that is strongly entangled with specific words. For example, when generating an image of "Charlie Chaplin", a "mustache" consistently appears even if explicitly instructed not to include it, as the concept of "mustache" is strongly entangled with "Charlie Chaplin". To address this issue, we propose a novel approach to directly suppress such entangled content within the text embedding space of diffusion models. Our method introduces a delta vector that modifies the text embedding to weaken the influence of undesired content in the generated image, and we further demonstrate that this delta vector can be easily obtained through a zero-shot approach. Furthermore, we propose a Selective Suppression with Delta Vector (SSDV) method to adapt the delta vector into the cross-attention mechanism, enabling more effective suppression of unwanted content in regions where it would otherwise be generated. Additionally, we enabled more precise suppression in personalized T2I models by optimizing the delta vector, which previous baselines were unable to achieve. Extensive experimental results demonstrate that our approach significantly outperforms existing methods, both in terms of quantitative and qualitative metrics.

## Method Overview

<div align="left">
    <img width="100%" alt="teaser" src="https://github.com/eunso999/SSDV/blob/main/source/method_overview.png?raw=true"/>
</div>

We introduce two methods for obtaining delta: a zero-shot approach that obtains delta without any additional training (top-left corner) and an optimization approach that yields a more precise delta (right side). In the bottom-left corner, we illustrate the Selective Suppression with Delta Vector, showing how the obtained delta vector is applied to the text embedding for content suppression and how it operates within the cross-attention layer.

## Installation

To set up the environment run the following command:

```bash
conda create -n SSDV python=3.10 -y
conda activate SSDV

pip install -r requirements.txt
```

## Suppression

To apply suppression with our method, please run the notebooks [suppression_sd.ipynb](https://github.com/eunso999/SSDV/blob/main/suppression_sd.ipynb) and [suppression_dreambooth.ipynb](https://github.com/eunso999/SSDV/blob/main/suppression_dreambooth.ipynb).

- The suppression_sd.ipynb notebook demonstrates suppression using the **zero-shot delta** approach on **Stable Diffusion v1.5**.
- The suppression_dreambooth.ipynb notebook demonstrates suppression on a **DreamBooth-tuned Stable Diffusion v1.5 model**, using both the **zero-shot** and **optimized delta** approaches.

For better results, adjust the 'ALPHA' value in the notebook.

## Checklist

- [x] Upload suppression code for Stable Diffusion
- [x] Upload suppression code for DreamBooth-tuned model
- [ ] Release SEP-Benchmark
- [ ] Update local blending code

## Citation

If our repository is helpful for your research, please consider to cite it:

```bibtex
@article{koh2025translation,
  title={Translation of Text Embedding via Delta Vector to Suppress Strongly Entangled Content in Text-to-Image Diffusion Models},
  author={Koh, Eunseo and Hong, Seunghoo and Kim, Tae-Young and Woo, Simon S and Heo, Jae-Pil},
  journal={arXiv preprint arXiv:2508.10407},
  year={2025}
}
```