/*
    Created by Netlab 22214414
    Last updated on 2022/9/22
*/
#include "Broadcaster.h"

// 根据输入视频源创建AVFormatContext
AVFormatContext* Broadcaster:: create_de_formatc( const std::string& inputsource ) {

    AVFormatContext* avfc = avformat_alloc_context();
    if ( !avfc ) {
        throw std::runtime_error( "failed to alloc memory for avformat" );
    }

    int ret = avformat_open_input(&avfc, inputsource.c_str(), nullptr, nullptr);
    if ( ret != 0 ) {
        throw std::runtime_error( "failed to open input file" );
    }

    ret = avformat_find_stream_info( avfc, nullptr );
    if (ret != 0) {
        throw std::runtime_error("failed to get stream info");
    }

    return avfc;
}   

// 创建输入视频源解码器相关信息
AVCodecContext* Broadcaster:: create_de_codecc( AVFormatContext* avformatc , int thread_num , int& video_index) {

    assert( avformatc != nullptr );
    AVStream* de_stream = nullptr;
    // find stream index
    for ( int i = 0; i < avformatc->nb_streams; i++ ) {
        if ( avformatc->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO ) {
            video_index = i;
            de_stream = avformatc->streams[i];
            break;
        }
    }
    // find de_codec by codec id
    AVCodec* de_codec =  avcodec_find_decoder( de_stream->codecpar->codec_id );
    if ( !de_codec ) {
        throw std::runtime_error("failed to find the de_codec");
    }
    // use the de_codec to create de_codec_context
    AVCodecContext* av_codecc = avcodec_alloc_context3( de_codec );
    if ( !av_codecc ) {
        throw std::runtime_error("failed to alloc memory for de_codec context");
    }
    // copy the params from de_stream to de_codec_context
    int ret = avcodec_parameters_to_context( av_codecc, de_stream->codecpar );
    if ( ret < 0 ) {
        throw std::runtime_error("failed to fill de_codec context");
    }
    // use this to speed up decode
    av_codecc->thread_count = thread_num;
    // open the de_codec using de_codec_context;
    ret = avcodec_open2( av_codecc, de_codec, nullptr);
    if ( ret < 0 ) {
        throw std::runtime_error("failed to open de_codec");
    }

    return av_codecc;

}

// 创建推流的低清视频解码器相关信息
AVCodecContext* Broadcaster:: create_lr_de_codecc() {

    AVCodec* de_codec = avcodec_find_decoder( AV_CODEC_ID_H264 );
    if ( !de_codec ) {
        throw std::runtime_error("failed to find the lr_de_codec");
    }

    AVCodecContext* av_codecc = avcodec_alloc_context3( de_codec );
    if ( !av_codecc ) {
        throw std::runtime_error("failed to alloc memory for lr_de_codec context");
    }

    //
    Broadcaster:: set_av_codecc( av_codecc );

    int ret = avcodec_open2( av_codecc, de_codec, nullptr );
    if ( ret < 0 ) {
        throw std::runtime_error("failed to open lr_de_codec");
    }

    return av_codecc;
}

// 创建并设置编码器参数
void Broadcaster:: create_en_params( const std::string& video_output_path ) {

    //alloc memory for en stream
    avformat_alloc_output_context2(& this->en_formatc, nullptr, "rtsp", video_output_path.c_str() );
    if ( !this->en_formatc ) {
        throw std::runtime_error("could not allocate memory for en_formatc");
    }

    // create codec as H264
    AVCodec* en_codec = avcodec_find_encoder( AV_CODEC_ID_H264 );
    if ( !en_codec ) {
        throw std::runtime_error("could not find the en_codec");
    }

    // using en_codec to create en_codecc
    this->en_codecc = avcodec_alloc_context3( en_codec );
    if ( !this->en_codecc ) {
        throw std::runtime_error("failed to alloc memory for en_codec context");
    }

    // setting en_codecc params
    Broadcaster:: set_av_codecc( this->en_codecc );

    // open the en_codec
    int ret = avcodec_open2( this->en_codecc, en_codec, nullptr );
    if ( ret < 0 ) {
        throw std::runtime_error("failed to open en_codec");
    }

    //create en_stream
    this->en_stream = avformat_new_stream( this->en_formatc, nullptr );

    ret = avcodec_parameters_from_context( this->en_stream->codecpar, this->en_codecc );
    if ( ret < 0 ) {
        throw std::runtime_error("failed to copy the en_codec params to en_stream");
    }

    //mamually set time_base ans fps of en_stream;
    this->en_stream->time_base = this->en_codecc->time_base;
    this->en_stream->avg_frame_rate = this->en_codecc->framerate;

}

// 设置编解码器的相关参数
void Broadcaster:: set_av_codecc( AVCodecContext* av_codecc ) {

    /*
        设置编解码器的一些参数

        *preset:    编码预设等级从低到高: ultrafast, superfast, veryfast, faster, fast, medium, slow, slower, veryslow, placebo
                    等级越高画面质量越好，编码越慢
        
        *profile    编码图像等级 baseline  main     high
        *tune        针对不同场景的优化选项 
        keyint=24:min-keyint=24:no-scenecut:
                    keyint: maximum length of the GOP, maximum interval between each keyframe
                    min-keyint: the minimum length of the GOP
                    scenecut: 当有场景切换时，会插入额外的I帧，可能会浪费带宽，一般关闭场景切换

        *height, width
        sample_aspect_ratio:    图像采集时，横向采集点数与纵向采集点数的比例

        *pix_fmts:      pixel format, 像素格式 yuv rgb等等

        *time_base:     时基  不可太小

        *framerate:     FPS

        *bit_rate:      码率    bps

        *rc_buffer_size:    编码器缓冲区大小

        rc_max_rate, rc_min_rate:   maximum bitrate, minmun bitrate

    */

    av_opt_set( av_codecc->priv_data, "preset", this->options.preset.c_str(), 0);

    av_codecc->height = this->de_codecc_hr->height / 2;
    av_codecc->width = this->de_codecc_hr->width / 2 ;

    av_codecc->pix_fmt = AV_PIX_FMT_YUV420P;

    av_codecc->time_base = {1, TIME_BASE};
    av_codecc->framerate = {FPS, 1};

    av_codecc->bit_rate = 1000 * this->options.bit_rate ; // kbps

    av_codecc->thread_count = 16;
}

// 初始化与推流相关的参数
void Broadcaster:: init_write_head( AVFormatContext* av_formatc, const std::string& video_output_path ) {
    
    // Format wants global header
    if ( av_formatc->oformat->flags & AVFMT_GLOBALHEADER ) {
        av_formatc->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
    }

    // Create and initialize a AVIOContext for accessing the resource indicated by url.
    if ( !(av_formatc->oformat->flags & AVFMT_NOFILE) ) {
        if (avio_open( &av_formatc->pb, video_output_path.c_str(), AVIO_FLAG_WRITE) < 0 ) {
            throw std::runtime_error("could not open the output file");
        }
    }

    //add params to write header, using tcp
    AVDictionary* opts = nullptr;
    av_dict_set( &opts, "rtsp_transport", "tcp", 0 );

    // write header
    int ret = avformat_write_header( av_formatc, &opts );
    if ( ret < 0 ) {
        throw std::runtime_error("an error occurred when opening output file");
    }

}

/*
    解码原始高清视频：
        读取原始高清视频流，得到相关的av_packet，将av_packet输入进高清视频解码器中解码得到av_frame视频帧
        根据相应的采样周期将相应的视频帧传入训练队列中，
        将得到的高清视频帧传入到高清解码队列中等待编码。
*/
void Broadcaster:: decoding_hr() {

    AVFrame* input_frame_hr = av_frame_alloc();
    AVPacket* input_packet_hr = av_packet_alloc();
    AVPacket* tmp_packet = nullptr;
    bool stop = false;
    int frame_number = 0;

    auto time1 = Utils::clock();

    while ( !stop ) {
        // read input packet
        int64_t tp_start = av_gettime();
        int ret = av_read_frame( this->de_formatc_hr, input_packet_hr );

        if ( ret < 0 ) {
            stop = true;
            // set tmp_packet nullptr to reflush decoder
            tmp_packet = nullptr;
        } else {
            // skip audio stream, just process video stream
            if ( input_packet_hr->stream_index != this->hr_video_index ) {
                continue;
            }
            // covert time base:  en_time = de_time * de_time_base / en_time_base;
            input_packet_hr->pts = av_rescale_q( input_packet_hr->pts, 
                                                this->de_formatc_hr->streams[this->hr_video_index]->time_base, {1, TIME_BASE} );
            input_packet_hr->dts = av_rescale_q( input_packet_hr->dts, 
                                                this->de_formatc_hr->streams[this->hr_video_index]->time_base, {1, TIME_BASE} );
            input_packet_hr->duration = av_rescale_q( input_packet_hr->duration,
                                                this->de_formatc_hr->streams[this->hr_video_index]->time_base, {1, TIME_BASE} );
            tmp_packet = input_packet_hr;
        }

        // decode packet to get frame
        ret = avcodec_send_packet( this->de_codecc_hr, tmp_packet );
        if (ret < 0) {
            std::cout << "Error while sending packet to decoder" << std::endl;
            return;
        }

        while ( ret >= 0 ) {
            // read decoded frame
            ret = avcodec_receive_frame( this->de_codecc_hr, input_frame_hr );
            if ( ret == AVERROR(EAGAIN) || ret == AVERROR_EOF ) {
                break;
            }else if (ret < 0) {
                std::cout<< "Error while receiving frame from decoder" << std::endl;
                return;
            }

            // if ( frame_number % 30 == 0 ) {
            //     cv::Mat tmp = Broadcaster::av_frame_yuv420p_to_cv_mat_bgr(input_frame_hr);
            //     可以在这里保存tmp得到原始的高清解码视频帧
            // }

            // sampling frame for model training
            if ( (frame_number % this->options.sample_interval == 0)  
                &&  (frame_number % this->options.training_cycle < this->options.sampling_cycle) ) {
                cv::Mat hr_mat = Broadcaster::av_frame_yuv420p_to_cv_mat_bgr(input_frame_hr);
                
                for ( int gpu_i = 0; gpu_i < this->gpu_count; gpu_i++ ) {
                    this->hr_mat_queue_list[gpu_i]->push( std::make_pair(hr_mat, frame_number) );
                }

            }

            // push decoded frame to queue
            this->decoded_frame_queue->push( av_frame_clone(input_frame_hr) );

            av_frame_unref( input_frame_hr );

            frame_number++;

            // delay to receive each frame
            int64_t tp_end = av_gettime();
            int64_t tp_duration = tp_end - tp_start;
            if ( tp_duration < 1000 * 1000 / FPS ) {
                av_usleep( ( 1000 * 1000 / FPS - tp_duration ) * 0.9 );
            }
            tp_start = tp_end;
            
        }   

        if ( tmp_packet ) {
            av_packet_unref( tmp_packet );
        }

    }

    auto time2 = Utils::clock();
    av_frame_free(&input_frame_hr);
    av_packet_free(&input_packet_hr);

    this->is_decode_hr_done = true;
    std::cout << "Decoding hr  done " << Utils::get_duration(time1, time2) << "in ms, with numbers: "<< frame_number << std::endl;
}

/*
    解码编码后的低清视频：
        为了保证SR的效果，训练时的模型输入和推理时的模型输入应该保持一致。
        故需要将得到的低清推流视频再次解码，得到的模型训练输入帧和播放端得到的模型推理帧才能保持一致
*/
void Broadcaster:: decoding_lr() {

    AVFrame* frame_lr = av_frame_alloc();
    int frame_number = 0;

    auto time1 = Utils::clock();
    while( !( this->is_encode_done && this->encoded_packet_for_dec_queue->empty() ) ) {

        if ( this->encoded_packet_for_dec_queue->empty() )  {
            continue;
        }

        AVPacket* lr_packet = nullptr;

        this->encoded_packet_for_dec_queue->pop( lr_packet );

        int ret = avcodec_send_packet( this->de_codecc_lr, lr_packet );
        if ( ret < 0 ) {
            std::cout << "Error while sending packet to lr decoder " << std::endl;
            return;
        }

        while( ret >= 0 ) {
            //read decoded frame
            ret = avcodec_receive_frame( this->de_codecc_lr, frame_lr );
            if ( ret == AVERROR(EAGAIN) || ret == AVERROR_EOF ) {
                break;
            }else if (ret < 0) {
                std::cout<< "Error while receiving frame from decoder" << std::endl;
                return;
            }

            if ( (frame_number % this->options.sample_interval == 0)  
                &&  (frame_number % this->options.training_cycle < this->options.sampling_cycle) ) {

                cv::Mat lr_mat = Broadcaster::av_frame_yuv420p_to_cv_mat_bgr( frame_lr );
                for ( int gpu_i = 0; gpu_i < this->gpu_count; gpu_i++ ) {
                    this->lr_mat_queue_list[gpu_i]->push( std::make_pair(lr_mat, frame_number) );
                }

            }

            av_frame_unref( frame_lr );

            frame_number++;

        }
        av_packet_free(&lr_packet);
    }   
    //flush 
    int ret = avcodec_send_packet( this->de_codecc_lr, nullptr );
    if ( ret < 0 ) {
        std::cout << "Error while sending packet to lr decoder " << std::endl;
        return;
    }
    while( ret >= 0 ) {
        //read decoded frame
        ret = avcodec_receive_frame( this->de_codecc_lr, frame_lr );
        if ( ret == AVERROR(EAGAIN) || ret == AVERROR_EOF ) {
            break;
        }else if (ret < 0) {
            std::cout<< "Error while receiving frame from decoder" << std::endl;
            return;
        }

        if ( (frame_number % this->options.sample_interval == 0)  
            &&  (frame_number % this->options.training_cycle < this->options.sampling_cycle) ) {

            cv::Mat lr_mat = Broadcaster::av_frame_yuv420p_to_cv_mat_bgr( frame_lr );
            
            for ( int gpu_i = 0; gpu_i < this->gpu_count; gpu_i++ ) {
                this->lr_mat_queue_list[gpu_i]->push( std::make_pair(lr_mat, frame_number) );
            }

        }

        av_frame_unref( frame_lr );

        frame_number++;

    }
    auto time2 = Utils::clock();
    av_frame_free(&frame_lr);

    this->is_decode_lr_done = true;
    std::cout << "Decoding lr done " << Utils::get_duration(time1, time2) << "in ms, with numbers: "<< frame_number << std::endl;
}

/*
    将高清视频帧进行缩小得到低清视频帧
    将低清视频帧编码成低清视频源供后续推流：

*/
void Broadcaster:: encoding() {

    auto time1 = Utils::clock();
    // stop while when decoding is done and decoded_frame_queue is empty
    while ( !( this->is_decode_hr_done && this->decoded_frame_queue->empty() ) ) {

        // waiting for the decoded_frame_queue
        if ( this->decoded_frame_queue->empty() ) {
            continue;
        }

        AVFrame* frame = nullptr;
        AVFrame* scaled_frame = av_frame_alloc();

        // get decoded frame 
        this->decoded_frame_queue->pop( frame );

        Broadcaster::scale_av_frame(frame, scaled_frame, frame->width / 2, frame->height / 2);
        av_frame_free(&frame);
        
        int ret =  Broadcaster::encode_frame_to_queue( scaled_frame );
        if ( ret < 0 ) {
            return;
        }

        // since we don't use ffmpeg fuctions to alloc memory, just using av_frame_free can't free the memory
        // we use memcpy to alloc memory, so we have to use av_freep to free the memory
        av_freep(&scaled_frame->data[0]);
        av_frame_free(&scaled_frame);

    }

    // flush encoder to output rest of frames
    int ret = Broadcaster::encode_frame_to_queue( nullptr );
    if ( ret < 0 ) {
        return;
    }

    auto time2 = Utils::clock();
    this->is_encode_done = true;
    std::cout<< "Encoding done " << Utils::get_duration(time1, time2) << " in ms"<< std::endl;

}

// 将原始高清帧进行缩放操作
void Broadcaster:: scale_av_frame( AVFrame* frame, AVFrame* output_frame, int width, int height ) {

    assert( frame != nullptr );

    // set output_frame
    output_frame->format = AV_PIX_FMT_YUV420P;
    output_frame->width = width;
    output_frame->height = height;
    output_frame->pts = frame->pts;
    output_frame->pict_type = frame->pict_type;

    //alloc memory for output_frame
    av_image_alloc( output_frame->data, output_frame->linesize, output_frame->width, output_frame->height,
                    static_cast<AVPixelFormat>(output_frame->format), 16);
    
    /*
        struct SwsContext *sws_getContext(
            int srcW,  // 源图像的宽度 
            int srcH, // 源图像的宽度 
            enum AVPixelFormat srcFormat, //源图像的像素格式 
            int dstW, // 目标图像的宽度 
            int dstH, // 目标图像的高度 
            enum AVPixelFormat dstFormat, // 目标图像的像素格式 
            int flags,//选择缩放算法(只有当源图像和目标图像大小不同时有效),一般选择SWS_FAST_BILINEAR 
            SwsFilter *srcFilter, //源图像的滤波器信息, 若不需要传nullptr 
            SwsFilter *dstFilter, // 目标图像的滤波器信息, 若不需要传nullptr 
            const double *param // 特定缩放算法需要的参数，默认为nullptr 
            );
    */
    
    SwsContext *sws_context = sws_getContext(frame->width, frame->height, AV_PIX_FMT_YUV420P,
                                                 output_frame->width, output_frame->height, AV_PIX_FMT_YUV420P,
                                                 SWS_BICUBIC, nullptr, nullptr, nullptr);

    sws_scale(sws_context, frame->data, frame->linesize, 0, frame->height, 
                            output_frame->data, output_frame->linesize);

    sws_freeContext( sws_context );
}

// 将视频帧编码成视频源
int Broadcaster:: encode_frame_to_queue( AVFrame* frame ) {

    if ( frame && frame->pict_type != AV_PICTURE_TYPE_I ) {
        frame->pict_type = AV_PICTURE_TYPE_NONE;
    }

    AVPacket* output_packet = av_packet_alloc();

    int ret = avcodec_send_frame( this->en_codecc, frame );

    while( ret >= 0 ) {
        // receive encoded packet
        ret = avcodec_receive_packet( this->en_codecc, output_packet );
        if ( ret == AVERROR(EAGAIN) || ret == AVERROR_EOF ) {
            break;
        } else if ( ret < 0 ) {
            std::cout << "Error while receiving packet from encoder" << std::endl;
            return ret;
        }

        //set encoded packet
        output_packet->stream_index = this->hr_video_index;
        output_packet->duration = this->en_stream->time_base.den / this->en_stream->time_base.num / FPS;
        output_packet->pts = av_rescale_q_rnd( output_packet->pts, {1, TIME_BASE}, this->en_stream->time_base, 
                                        (AVRounding)(AV_ROUND_NEAR_INF | AV_ROUND_PASS_MINMAX) );
        output_packet->dts = av_rescale_q_rnd( output_packet->dts, {1, TIME_BASE}, this->en_stream->time_base, 
                                        (AVRounding)(AV_ROUND_NEAR_INF | AV_ROUND_PASS_MINMAX) );

        // push encoded packet into encoded_packet_queue
        this->encoded_packet_queue->push( av_packet_clone( output_packet) );

        this->encoded_packet_for_dec_queue->push( av_packet_clone( output_packet) );

        av_packet_unref( output_packet );

    }
    av_packet_free(&output_packet);

    return 0;
}

/*
    将得到的编码后低清视频推流到RTSP服务器
    进行SR模型的在线训练
*/
void Broadcaster:: broadcasting() {

    int frame_sent_count = 0;
    auto time1 = Utils::clock();
    std::thread online_training_thread = std::thread(&Broadcaster::online_training, this, frame_sent_count);

    // stop while when encoding is done and encoded_packet_queue is empty
    while( !( this->is_encode_done && this->encoded_packet_queue->empty() ) ) {
        
        // waiting for the encoded_packet_queue;
        if ( this->encoded_packet_queue->empty() ) {
            continue;
        }

        AVPacket* packet = nullptr;

        // get encoded packet
        this->encoded_packet_queue->pop( packet );
        
        // write packet 
        int ret = av_interleaved_write_frame( this->en_formatc, packet );
        if ( ret != 0 ) {
            std::cout << "Error while sending packet!" << std::endl;
            av_packet_free(&packet);
            return;
        }
        av_packet_free(&packet);
        frame_sent_count++;
        //Online training
        if ( frame_sent_count % this->options.sample_interval == 0 ) {
            std::cout << "broadcaster:" << frame_sent_count << std::endl;
        }
        if ( frame_sent_count % this->options.training_cycle == 0 ) {
            std::cout <<"waiting for the lastest model trained and sent!"<< std::endl;
            online_training_thread.join();

            std::cout << "training for the next time slot!" << std::endl;
            online_training_thread = std::thread(&Broadcaster::online_training, this, frame_sent_count);

        }

    }

    av_write_trailer( this->en_formatc );

    auto time2 = Utils::clock();
    this->is_broadcast_done = true;
    online_training_thread.join();
    std::cout<< "Broadcaster::broadcasting done "<<Utils::get_duration(time1, time2) << " ms, with numbers "<< frame_sent_count << std::endl;
}

/*
    在线训练，得到相应的SR模型后使用Post请求直接传递到播放端的http服务器上
*/
void Broadcaster:: online_training( int frame_number ) {

    bool ret = this->trainer->training( this->lr_mat_queue_list, this->hr_mat_queue_list );
    if ( ret ) {
        httplib::Client client( this->options.http_server );
        httplib::MultipartFormDataItems items = {
            {"model", this->trainer->get_model_data(), "", "text/plain"},
            {"Ack", std::to_string(frame_number), "", "text/plain"}
        };
        auto response = client.Post("/model", items);

        if ( response->status != 200 ) {
            std::cout<< "http Post model return " << response->status << std::endl;
        }

    } else {
        std::cout << "no model to update!" << std::endl;
    }
}

/*  
    推流端的起始函数：
    串行逻辑: 原始的高清视频源 ---(decoding_hr)---> 高清视频帧 ---( encoding )---> 低清视频源 ---( decoding_lr )---> 低清视频帧\
             ---(Online training)---> SR模型 ---(broadcasting)---> 推流相应的低清视频和SR模型
    为保证实时性，并行实现这些操作，每个函数之间通过单生产者单消费者的无锁队列进行数据传递
*/
void Broadcaster:: start() {

    std::cout << "Broadcasting start!" << std::endl;

    std::thread decoding_hr_thread(&Broadcaster::decoding_hr, this);
    
    std::thread encoding_thread(&Broadcaster::encoding, this);

    std::thread decoding_lr_thread(&Broadcaster::decoding_lr, this);

    std::thread broadcasting_thread(&Broadcaster::broadcasting, this);

    decoding_hr_thread.join();

    encoding_thread.join();

    decoding_lr_thread.join();

    broadcasting_thread.join();

    assert( this->is_broadcast_done == true );

    std::cout << "Waiting for http_serving_thread" << std::endl;
    std::string in;
    std::cin >> in;

    std::cout << "Broadcasting done!" << std::endl;

}

// 将av_frame格式的视频帧转换为cv Mat bgr格式
cv::Mat Broadcaster:: av_frame_yuv420p_to_cv_mat_bgr(AVFrame* frame) {
        int width = frame->width;
        int height = frame->height;
        // cv mat yuv format: only one channel
        /*             <------------Y-linesize----------->
         *             <-------------width------------>
         *             -----------------------------------
         *             |                              |  |
         *             |                              |  |
         *   height    |              Y               |  |
         *             |                              |  |
         *             |                              |  |
         *             |                              |  |
         *             -----------------------------------
         *             |             |  |             |  |
         * height / 2  |      U      |  |      V      |  |
         *             |             |  |             |  |
         *             -----------------------------------
         *             <---U-linesize--> <--V-linesize--->
         *             <---U-width--->   <--V-width--->
         */
        cv::Mat img = cv::Mat::zeros(height * 3 / 2, width, CV_8UC1);

        for (int j = 0; j < height; j++) {
            memcpy(img.data + j * width, frame->data[0] + j * frame->linesize[0], width);
        }
        for (int j = 0; j < height / 2; j++) {
            memcpy(img.data + height * width + j * width / 2, frame->data[1] + j * frame->linesize[1], width / 2);
        }
        for (int j = 0; j < height / 2; j++) {
            memcpy(img.data + height * width * 5 / 4 + j * width / 2, frame->data[2] + j * frame->linesize[2],
                   width / 2);
        }

        cv::Mat rgb;
        cvtColor(img, rgb, cv::COLOR_YUV2BGR_I420);
        return rgb;
}

// 返回相应的训练用的cuda编号
std::string Broadcaster:: get_cuda_num( int idx ) {
    return  "cuda:" + std::to_string( this->options.cuda_num[idx] ) ;
}