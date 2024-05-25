# LLama3工具集

本工具集包括了LLama3模型的安装、Web应用、微调程序、Openai API开发与客户端应用以及使用Meta官方微调程序进行微调的步骤。

## 一、准备工作

GPU：16G以上，建议24G

操作系统：建议Ubuntu22.04

CUDA：12.0及以上

Anaconda：最新版

## 二、模型部署

### 1、安装依赖环境

```bash
# 1、clone代码
git clone https://github.com/little51/llama3-tools
cd llama3-tools/
# 2、创建虚拟环境
conda create -n llama3 python=3.10 -y
conda activate llama3
# 3、安装依赖库
pip install -r requirements.txt -i https://pypi.mirrors.ustc.edu.cn/simple
# 4、验证PyTorch
python -c "import torch; print(torch.cuda.is_available())"
```

### 2、PyTorch修正（如有问题）

```bash
# 如果验证PyTorch返回False或报错，则按以下步骤重装PyTorch
# 1、卸载
pip uninstall torch -y 
pip uninstall torchvision -y
# 2、安装旧版本torch
# 如CUDA 12.0
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 \
--index-url https://download.pytorch.org/whl/cu121
# 如CUDA 11.7
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 \
-i https://pypi.mirrors.ustc.edu.cn/simple 
# 3、验证
python -c "import torch; print(torch.cuda.is_available())"
```

### 3、下载模型

```bash
# 模型下载脚本从aliendao.cn首页下载
# 链接为 https://aliendao.cn/model_download.py
# linux下使用wget命令下载，windows下直接在浏览器打开链接下载
wget https://aliendao.cn/model_download.py
python model_download.py --repo_id NousResearch/Meta-Llama-3-8B-Instruct
```

### 4、运行WEB测试程序

```bash
python llama3-gradio.py
# 访问http://服务器IP:6006/
```
## 三、Openai API开发

### 1、运行API服务程序

```bash
python llama3-api.py
```

### 2、运行客户端程序

```bash
# 需要先安装node.js
cd chat-app
npm i
# 修改src/app.js第12行的baseURL为：http://服务器IP:6006/v1
npm start
# 访问http://127.0.0.1:3000/
```
## 四、模型微调

```bash
# 微调
python llama3-train.py
# 模型合并
python merge_lora_weights.py \
--base_model ./dataroot/models/NousResearch/Meta-Llama-3-8B-Instruct \
--peft_model output/PEFT/model \
--output_dir output/merged/model
# 合并后，就可以把output/merged/model下的模型装载推理了
```

## 五、使用llama-recipes微调

### 1、微调环境安装

```bash
# CUDA：2.0.1，GPU：16G以上
# 下载llama-recipes最新源码
git clone https://github.com/meta-llama/llama-recipes
cd llama-recipes
git checkout cf29a56
# 以源码安装llama-recipes
conda deactivate
conda create -n llama-recipes python=3.10 -y
conda activate llama-recipes
pip install -U pip setuptools -i https://pypi.mirrors.ustc.edu.cn/simple
pip install -e . -i https://pypi.mirrors.ustc.edu.cn/simple
# 验证PyTorch
python -c "import torch; print(torch.cuda.is_available())"
```

### 2、数据准备

```bash
# alpaca_data.json文件复制到llama-recipes/src/llama_recipes/datasets目录下
# alpaca_data.json数据条数不能太少，防止在微调时出现
# ZeroDivisionError: float division by zero错误
```

### 3、微调过程

```bash
# 微调(更详细的微调参数见：
# https://github.com/meta-llama/llama-recipes/tree/main/recipes/finetuning)
python -m llama_recipes.finetuning \
--use_peft \
--dataset alpaca_dataset \
--peft_method lora \
--batch_size_training 1 \
--model_name ../dataroot/models/NousResearch/Meta-Llama-3-8B-Instruct \
--output_dir output/PEFT/model \
--quantization
```

### 4、合并原始模型和Lora微调模型

```bash
python merge_lora_weights.py \
--base_model ../dataroot/models/NousResearch/Meta-Llama-3-8B-Instruct \
--peft_model output/PEFT/model \
--output_dir output/merged/model
```