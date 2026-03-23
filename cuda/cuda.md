# nvidia-smi
```shell
nvidia-smi --id=0 --query-gpu=index,uuid,name,timestamp,memory.total,memory.free,memory.used,utilization.gpu,utilization.memory --format=csv,noheader%s -lms 50
```