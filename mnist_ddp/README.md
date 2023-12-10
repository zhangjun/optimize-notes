# torchrun
```python
export NCCL_DEBUG=INFO
python -c "import torch; print(torch.version.cuda)"

torchrun --standalone --nproc_per_node=4  datastet_info.py
torchrun --nnodes=1 --nproc_per_node=4  datastet_info.py
```
- multi-instances on one node specified by the rdzv-endpoint port
```python
torchrun
    --nnodes=$NUM_NODES
    --nproc-per-node=$NUM_TRAINERS
    --max-restarts=3
    --rdzv-id=$JOB_ID
    --rdzv-backend=c10d
    --rdzv-endpoint=$HOST_NODE_ADDR
    YOUR_TRAINING_SCRIPT.py (--arg1 ... train script args...)
```
If no port number is specified HOST_NODE_ADDR defaults to 29400.


