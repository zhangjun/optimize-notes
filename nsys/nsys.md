# install
```shell
apt update
apt install -y --no-install-recommends gnupg
echo "deb http://developer.download.nvidia.com/devtools/repos/ubuntu$(source /etc/lsb-release; echo "$DISTRIB_RELEASE" | tr -d .)/$(dpkg --print-architecture) /" | tee /etc/apt/sources.list.d/nvidia-devtools.list
apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
apt update
apt install nsight-systems-cli
```

# usage
- prof service
```shell
nsys launch --trace=cublas,cuda,cudnn,nvtx --show-output=true --cuda-memory-usage=true --cuda-graph-trace=graph --trace-fork-before-exec=true --session-new vllm --stats true  ${launch_cmd}

nsys start --stats=true --output output_nsys --session vllm -f true

nsys stop --session vllm
```
