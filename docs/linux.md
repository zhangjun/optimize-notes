# Linux setup v2ray client
- [linux下配置V2ray作为客户端来访问GitHub、G*le等服务](https://www.witersen.com/?p=1408)
```shell
wget https://github.com/v2fly/v2ray-core/releases/download/v4.31.0/v2ray-linux-64.zip
v2ray -test -config config.json
v2ray –config=config.json
curl –socks5 127.0.0.1:1080 https://www.google.com
```