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
# pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
# conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.4 -c pytorch -c nvidia
# pip install mkl==2024.0.0
pip install --index-url https://download.pytorch.org/whl/cu124 torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1

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

pip install scikit-image
```

* 修补cuda核 (PS:这部分可能存在较多的bug，此处的setup.py代码也是经过修复的~)

```bash
cd src/croco/models/curope/
# rm -rf build/
python setup.py build_ext --inplace
cd ../../../../

# 若出现cuda不匹配问题，需要在.bashrc中设置一下~
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

下载所有的模型以及checkpoints,注意需要先到[网站](https://smpl.is.tue.mpg.de)以及[网站2](https://download.is.tue.mpg.de/download.php?domain=smplx&sfile=models_smplx_v1_1.zip)进行注册；

此外，还需要科学上网才可以下载到google drive里面的东西 `pip install gdown`

```bash
# SMPLX family models
bash scripts/fetch_smplx.sh

# Human3R checkpoints
huggingface-cli download faneggg/human3r human3r.pth --local-dir ./src
# 此处的下载建议用脚本
```

* 采用脚本下载huggingface：`python download_huggingface.py`

```py
from huggingface_hub import snapshot_download  # 注意这里导入了 hf_hub_download

model_path = snapshot_download(
    repo_id="faneggg/human3r",
    local_dir="/home/guanweipeng/Human3R/huggingface_model",
    cache_dir="/home/guanweipeng/Human3R/huggingface_model/cache",  # 指定缓存目录
    token="hf_******",     # ✅ 在这里传 token
    endpoint="https://hf-mirror.com"   # 如果需要走镜像
)

print("文件下载到本地路径:", model_path)
```

~~~
cp /home/guanweipeng/Human3R/huggingface_model/human3r.pth ./src
~~~

* 数据集下载：

```bash
wget --post-data "username=wpguan@connect.hku.hk&password=1104672297a" 'https://download.is.tue.mpg.de/download.php?domain=rich&resume=1&sfile=test_hsc.zip' -O 'examples/test_hsc.zip' --no-check-certificate --continue
```

* 运行的时候可能遇到加载dinov2失败的情况：
  * 首先下载模型，应该是会存放在`/home/guanweipeng/.cache/torch/hub/main.zip`，运行`ls ~/.cache/torch/hub/facebookresearch_dinov2_main/`可以查看
  * 然后重新运行，运行前添加：`export TORCH_HOME=/home/guanweipeng/.cache/torch`

```bash
python
import torch
import torch.hub

# 手动下载 dinov2
model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14', pretrained=True, force_reload=True)

```

## 推理测试的demo

```bash
# input can be a folder or a video
# the following script will run inference with Human3R and visualize the output with viser on port 8080
CUDA_VISIBLE_DEVICES=0 python demo.py --model_path MODEL_PATH --size 512 \
    --seq_path SEQ_PATH --output_dir OUT_DIR --subsample 1 --use_ttt3r \
    --vis_threshold 2 --downsample_factor 1 --reset_interval 100

# Example:
CUDA_VISIBLE_DEVICES=0 python demo.py --model_path src/human3r.pth --size 512 --seq_path examples/GoodMornin1.mp4 --subsample 1 --use_ttt3r --vis_threshold 2 --downsample_factor 1 --reset_interval 100 --output_dir tmp

# CUDA_VISIBLE_DEVICES=0 python demo.py --model_path src/human3r.pth --size 512 --seq_path examples/boy-walking2.mp4 --subsample 1 --use_ttt3r --vis_threshold 2 --downsample_factor 1 --reset_interval 100 --output_dir tmp
```

结果会存放在`output_dir`文件夹内（但并没有看到~）