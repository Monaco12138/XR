<img src="https://github.com/Monaco12138/XR/blob/main/photo/framework3.png" width="100%">  

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

## 文件代码目录
#### 推流端
```c
|-- build  
|-- model  
|   |--  pretrain model  
|-- CMakeLists.txt  
|-- option.toml             \\配置文件
|-- Utils.h
|-- Setting.h               \\读取配置文件
|-- Broadcaster.h           \\定义推流端类
|-- Broadcaster.cpp         \\推流端类的实现
|-- main_broadcaster.cpp    \\推流端入口
|-- OnlineTraining.h        \\在线训练类及其实现
|-- PatchImgDataset.h       \\训练数据集设置
```

#### 播放端
```c
|-- build  
|-- model  
|   |--  pretrain model  
|-- CMakeLists.txt
|-- option.toml             \\配置文件
|-- Utils.h
|-- Setting.h               \\读取配置文件
|-- Streamplayer.h          \\定义播放器类
|-- main_streamplayer.cpp   \\播放端入口
|-- Streamplayer.cpp        \\播放器类的实现
```
## 基本使用
#### 编译
1. 以推流端为例，先进入./build目录
2. 执行命令
    ```
    cmake ..
    cmake --build . --config Release
    ```
3. 生成可执行文件即可运行./demo_broadcaster
4. 可随时修改配置文件option.toml里的内容来进行不同的设置，不需要再次编译

#### 运行
运行时将推流端和RTSP server 部署在A服务器上，播放端部署在B服务器上
* 推流端：  
    1. 运行前先确保RTSP server已经打开，在配置文件中设置好原始高清视频的路径，预训练模型保存的路径等相关路径确保无误
    2. 在配置文件中设置好可使用的cuda编号和想要推流的视频编码码率和编码质量，其余有关训练的设置可使用默认值
    3. 进入推流端的./build文件中运行./demo_broadcaster即可
    4. 推流端运行时为了保证客户端能完整收到所有的推流帧，要等待客户端初始化完毕后才开始推流，这里我们手动设置等待，到客户端初始化完毕后，随便输入一个字符串即可开启推流端的推流过程
