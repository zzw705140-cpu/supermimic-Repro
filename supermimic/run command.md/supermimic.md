# 评估
conda activate supermimic
cd "/home/xinyu/project/model-based RL/supermimic-Repro"
bash supermimic/workflows/eval.sh


# 训练
bash supermimic/workflows/train.sh


# GPU监控指令

watch -n 1 nvidia-smi