# Tensorrt-LLM
## Install 
```shell
apt-get -y install libopenmpi-dev && pip3 install --upgrade pip setuptools && pip3 install tensorrt_llm
```
## Build Model
```shell
python trtllm_build.py --build
```
## Serve Model
```shell
bash serve.sh
```
