<img src="https://github.com/Monaco12138/XR/blob/main/photo/framework3.png" width="100%">  

# 实时超分辨率传输样机实现

## 环境安装与部署

#### 语言和基础环境
* c++ 11
* ubuntu 20.04
* gcc 9.4.0
* cmake 3.16.3

#### SR模型训练推理相关环境
* cuda11.3.0: 需要去[NVIDIA官网](https://developer.nvidia.com/cuda-toolkit-archive)下载cuda11.3，选择runfile版本
* [cudnn v8.2.1](https://developer.nvidia.com/rdp/cudnn-archive):  下载完毕后根据[指导](https://docs.nvidia.com/deeplearning/cudnn/archives/index.html)安装即可，老版本要下载3个deb包
* [libtorch](https://pytorch.org/get-started/locally/) (cxx11 ABI) 对应cuda11.3版本，注意下载时要下stable-> linux-> libtorch-> c++/java-> cuda11.3-> cxx11 ABI版本
* [opencv-4.x](https://docs.opencv.org/4.x/d7/d9f/tutorial_linux_install.html) 下载完后需要手动编译安装

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
* [rtsp-simple-server](https://github.com/aler9/rtsp-simple-server)：一个零依赖的即用型服务代理，允许用户通过多种协议发布，读取和代
理实时的音视频流  
    1. 建议修改配置文件 rtsp-simple-server.yml 的默认配置，因为默认配置设置缓存及处理时间太小，当推流端推流太快时，会造成包丢失，而接收端若发现对应的包含I帧的包丢失，则会将这整个BOP的包全部丢弃，直到接收下一个I帧为止。这样可能造成的问题有：  1. 接收端在发送端未结束时就提前结束，2. 接收端得到的画面卡顿，跳跃。
    2. 修改默认配置如下:
        ```python
        # Timeout of read operations.
        readTimeout: 3600s
        # Timeout of write operations.
        writeTimeout: 3600s
        # Number of read buffers.
        # A higher number allows a wider throughput, a lower number allows to save RAM.
        readBufferCount: 2048
        ```
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
    1. 运行前先确保RTSP server已经打开，在配置文件中设置好原始高清视频的路径，预训练模型保存的路径等相关路径确保无误，http服务器请求路径为B服务器的ip，请求端口和B服务器监听端口要一致
    2. 在配置文件中设置好可使用的cuda编号和想要推流的视频编码码率和编码质量，其余有关训练的设置可使用默认值
    3. 进入推流端的./build文件中运行./demo_broadcaster即可
    4. 推流端运行时为了保证客户端能完整收到所有的推流帧，要等待客户端初始化完毕后才开始推流，这里我们手动设置等待，到客户端初始化完毕后，随便输入一个字符串即可开启推流端的推流过程

* 客户端：  
    1. 运行前要确保推流端已经运行，在配置文件中设置好相关的RTSP 服务器地址，预训练模型保存位置等
    2. 为保证实时推理，配置文件中需要设置好可用的cuda编号为4个gpu编号，注意推理时没有对这个做适配，需确保数量恰好为4
    3. 设置相应的参数, is_show为true表示实时播放推流画面， is_save_video为true表示存储相应的推流视频，存储位置为video_output_path的路径。注意因为再次编码存储推流视频会有一定的时间开销，这两个都设置为true时播放推流画面可能会有一定的卡顿。其余参数信息见配置文件注释。
    4. 设置完成后进入播放端的./build文件中运行./demo_streamplayer即可
    5. 播放端会等待推流端开始推流，此时推流端也等待播放端初始完成，我们在推流端随便输入一个字符串即可开启推流过程。
