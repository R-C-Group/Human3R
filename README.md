 <h1 align="center"> Human3R复现
  </h1>

[comment]: <> (  <h2 align="center">PAPER</h2>)
  <h3 align="center">
  <a href="https://github.com/fanegg/Human3R">Github</a>
  |<a href="https://fanegg.github.io/Human3R/">Website</a>
  | <a href="https://arxiv.org/pdf/2510.06219?">Paper</a>
  </h3>
  <div align="center"></div>

<br>

## 安装配置

```bash
git clone git@github.com:R-C-Group/Human3R.git

#创建环境
conda create -n human3r666 python=3.11 cmake
conda activate human3r666
# conda install pytorch torchvision pytorch-cuda=12.4 -c pytorch -c nvidia  # use the correct version of cuda for your system
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.4 -c pytorch -c nvidia
pip install mkl==2024.0.0

pip install -r requirements.txt
# pip install --no-build-isolation chumpy
pip install chumpy

# issues with pytorch dataloader, see https://github.com/pytorch/pytorch/issues/99625
conda install 'llvm-openmp<16'
# for training logging
conda install -y gcc_linux-64 gxx_linux-64

# pip install git+https://github.com/nerfstudio-project/gsplat.git
pip install gsplat

# for evaluation
pip install evo
pip install open3d
```

* 修补cuda核

```bash
cd src/croco/models/curope/
# rm -rf build/
python setup.py build_ext --inplace
cd ../../../../
```

下载所有的模型以及checkpoints：

```bash
# SMPLX family models
bash scripts/fetch_smplx.sh

# Human3R checkpoints
huggingface-cli download faneggg/human3r666 human3r666.pth --local-dir ./src
```
