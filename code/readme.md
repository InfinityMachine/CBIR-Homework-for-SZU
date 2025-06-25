> 2022080182 曹博宇 多媒体系统导论 2025 Lab5

### 目录结构

```perl
project_root/
├─ image_retrieval_system_cpu.py  # 主程序
├─ gui_retrieval.py         # GUI
├─ Holidays/
│   ├─ jpg/                 # 1 491 张图片
│   └─ lists.txt            # 官方分组文件(在提交文件中已经提供)
└─ Oxford5k/
    ├─ jpg/                 # 5 062 张图片
    └─ gt/                  # *_query/good/ok/junk.txt

```

### 运行环境

```bash
# ① 创建独立虚拟环境
conda create -n img_retrieval python=3.10 -y
conda activate img_retrieval

# ② 安装依赖
conda install -c conda-forge faiss-cpu -y      # FAISS
pip install torch torchvision pillow tqdm      # PyTorch 2.x+
# Tkinter 随 CPython 自带，无需额外安装

```

### 使用方法

```bash
python image_retrieval_system_cpu.py build-index  # 构建索引

python image_retrieval_system_cpu.py evaluate # 评测 mAP

python image_retrieval_system_cpu.py query \   # 获取 top 5 查询结果
       --img examples/000012.jpg \
       --top 5

# 或者可用 GUI 查询

```

