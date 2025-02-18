<div align="center">
  <h1 style="display: inline-block; margin: 0;">🚀Stop Looking for Important Tokens in Multimodal Language Models: Duplication Matters More</h1>
</div>

<h4 align="center"> 

[Zichen Wen](https://scholar.google.com/citations?user=N-aPFvEAAAAJ&hl=zh-CN)<sup>1,2</sup>,
Yifeng Gao<sup>1</sup>,
[Shaobo Wang](https://gszfwsb.github.io/)<sup>1</sup>,
[Junyuan Zhang](https://scholar.google.com/citations?user=uwwqEg8AAAAJ&hl=en)<sup>2</sup>,
Qintong Zhang<sup>2,4</sup>, <br>
[Weijia Li](https://liweijia.github.io/)<sup>3,2</sup>,
[Conghui He](https://conghui.github.io/)<sup>2✉</sup>,
[Linfeng Zhang](http://www.zhanglinfeng.tech/)<sup>1✉</sup>,


<sup>1</sup>Shanghai Jiao Tong University, <sup>2</sup>Shanghai AI Laboratory, <br>
<sup>3</sup>Sun Yat-sen University, <sup>4</sup>Peking University

</h4>

<div align="center">

[![arXiv](https://img.shields.io/badge/Arxiv-2502.11494-AD1C18.svg?logo=arXiv)](https://arxiv.org/pdf/2502.11494) 
</div>


## 👀 Overview


## 🛠 Preparation

1. Clone this repository.

```bash
git clone https://github.com/ZichenWen1/DART
cd DART
```

2. Environment Setup and Preparation

```Shell
 conda create -n DART python=3.10 -y
 conda activate DART
 pip install -e .
 pip install flash-attn --no-build-isolation
```

3. Download Multimodal Benchmark

Please follow the detailed instruction in [LLaVA-Evaluation](https://github.com/haotian-liu/LLaVA/blob/main/docs/Evaluation.md).

## 🎯 Usage
### 📖 Script Templates
```shell
bash scripts/v1_5/eval/[Benchmark].sh [Reduction_Ratio] [Max_Num_Trunction]
```

### 🐝 Examples
```Shell
CUDA_VISIBLE_DEVICES=0 bash scripts/v1_5/eval/textvqa.sh 0.778 128
```

```Shell
CUDA_VISIBLE_DEVICES=0 bash scripts/v1_5/eval/pope.sh 0.778 128
```

```Shell
CUDA_VISIBLE_DEVICES=0 bash scripts/v1_5/eval/mme.sh 0.778 128
```


## 🔑 License

This project is released under the [Apache 2.0 license](LICENSE).



## 👍 Acknowledgment
We extend our gratitude to the open-source efforts of [LLaVA](https://github.com/haotian-liu/LLaVA).


## 📩 Contact
For any question about our paper or code, please email `zichen.wen@outlook.com`.

