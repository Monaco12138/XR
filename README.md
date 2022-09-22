# 实时超分辨率传输样机实现

## 环境安装与部署

#### 语言和基础环境
* c++ 11
* ubuntu 20.04
* gcc 9.4.0
* cmake 3.16.3

#### SR模型训练推理相关环境
* cuda11.3: 需要去NVIDIA官网下载cuda11.3，选择runfile版本
* cudnn v8.2.1:  下载完毕后根据指导安装即可，老版本要下载3个deb包
* libtorch (cxx11 ABI) 对应cuda11.3版本
* opencv-4.x 下载完后需要手动编译安装

#### 视频编解码相关环境
* libavcodec
* libavformat
* libswresample
* libavutil
* libswscale

#### 第三方开源库
* boost/lockfree/spsc_queue.hpp: 引入支持单个生产者和单个消费者的无锁队列
* [httplib.h](https://github.com/yhirose/cpp-httplib): http 请求服务
* [toml11](https://github.com/ToruNiina/toml11): 配置文件的读取
* [rtsp-simple-server](https://github.com/aler9/rtsp-simple-server)：一个零依赖的即用型服务代理，允许用户通过多种协议发布，读取和代理实时的音视频流
