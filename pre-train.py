import os
import re
from datetime import datetime

# =====================SETTING=====================
CACHE_ROOT = './cache'
NUM_GPUS = 1
INPUT_MODEL = 'EleutherAI/pythia-2.8b'
MODEL_NAME = 'dolly'
LOCAL_TRAINING_ROOT = './train_result'
EXPERIMENT_ID = '1'
DBFS_OUTPUT_ROOT = './dbfs/dolly_training'
# =====================SETTING=====================

cache_root = './cache'
os.environ["TRANSFORMERS_CACHE"] = cache_root

try:
    from training.trainer import load_training_dataset, load_tokenizer

    # Cache data and tokenizer locally before creating a bunch of deepspeed processes and make sure they succeeds.
    load_training_dataset()
    load_tokenizer()
except ImportError:
    print("pre load_training_dataset failed, check if you have ./training/trainer.py file.")


timestamp = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
model_name = MODEL_NAME
experiment_id = EXPERIMENT_ID

input_model = INPUT_MODEL

model_scale = 12
if input_model.lower().endswith('2.8b') or input_model.lower().endswith('3b'):
    model_scale = 3
elif input_model.lower().endswith('6.9b') or input_model.lower().endswith('7b'):
    model_scale = 7

if experiment_id:
    experiment_id = re.sub(r"\s+", "_", experiment_id.strip())
    model_name = f"{model_name}__{experiment_id}"

checkpoint_dir_name = f"{model_name}__{timestamp}"


root_path = os.getcwd()
if model_scale >= 12:
    deepspeed_config = os.path.join(root_path, "config/ds_z3_bf16_config.json")
else:
    deepspeed_config = os.path.join(root_path, "config/ds_z3_bf16_config_cpu.json")

dolly_training_dir_name = "dolly_training"

local_training_root = LOCAL_TRAINING_ROOT


dbfs_output_root = None
if not dbfs_output_root:
    dbfs_output_root = f"./dbfs/{dolly_training_dir_name}"

os.makedirs(local_training_root, exist_ok=True)
os.makedirs(dbfs_output_root, exist_ok=True)
os.makedirs(cache_root, exist_ok=True)

local_output_dir = os.path.join(local_training_root, checkpoint_dir_name)
dbfs_output_dir = os.path.join(dbfs_output_root, checkpoint_dir_name)

num_gpus_flag = f"--num_gpus={NUM_GPUS}"

tensorboard_display_dir = f"{local_output_dir}/runs"
print('============================================')
print(f"Local Output Dir: {local_output_dir}")
print(f"DBFS Output Dir: {dbfs_output_dir}")
print(f"Tensorboard Display Dir: {tensorboard_display_dir}")
print('============================================')

env = f"TRANSFORMERS_CACHE='{cache_root}' TOKENIZERS_PARALLELISM=false"

batch_size = 6 if model_scale >= 12 else 3 if model_scale >= 7 else 2

print(f"{env} deepspeed {num_gpus_flag} \\\n\
    --module training.trainer \\\n\
    --input-model {input_model} \\\n\
    --deepspeed {deepspeed_config} \\\n\
    --epochs 2 \\\n\
    --local-output-dir {local_output_dir} \\\n\
    --dbfs-output-dir {dbfs_output_dir} \\\n\
    --per-device-train-batch-size {batch_size} \\\n\
    --per-device-eval-batch-size {batch_size} \\\n\
    --logging-steps 10 \\\n\
    --save-steps 200 \\\n\
    --save-total-limit 20 \\\n\
    --eval-steps 50 \\\n\
    --warmup-steps 50 \\\n\
    --test-size 200 \\\n\
    --lr 5e-6")
