# Duoli

机器翻译[Databricks dolly数据集](https://github.com/databrickslabs/dolly), 训练。

# How to Use
```bash
git clone https://github.com/databrickslabs/dolly.git
git clone https://github.com/VendaCino/Duoli.git
cp -r Duoli/* dolly/
cd dolly
pip install -r requirements.txt 
python pre-train.py
```
之后，执行控制台输出的指令

```bash
TRANSFORMERS_CACHE='./cache' TOKENIZERS_PARALLELISM=false deepspeed --num_gpus=1 \
    --module training.trainer \
    --input-model EleutherAI/pythia-2.8b \
    --deepspeed G:\CodeOther\Duoli\config/ds_z3_bf16_config_cpu.json \
    --epochs 2 \
    --local-output-dir ./train_result\dolly__1__2023-04-21T11:58:57 \
    --dbfs-output-dir ./dbfs/dolly_training\dolly__1__2023-04-21T11:58:57 \
    --per-device-train-batch-size 2 \
    --per-device-eval-batch-size 2 \
    --logging-steps 10 \
    --save-steps 200 \
    --save-total-limit 20 \
    --eval-steps 50 \
    --warmup-steps 50 \
    --test-size 200 \
    --lr 5e-6
```

修改 pre-train.py 文件中文件夹和GPU数量的配置
```py
# change pre-train.py SETTING
# =====================SETTING=====================
CACHE_ROOT = './cache'
NUM_GPUS = 1
INPUT_MODEL = 'EleutherAI/pythia-2.8b'
MODEL_NAME = 'dolly'
LOCAL_TRAINING_ROOT = './train_result'
EXPERIMENT_ID = '1'
DBFS_OUTPUT_ROOT = './dbfs/dolly_training'
# =====================SETTING=====================
```

