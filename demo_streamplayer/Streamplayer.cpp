/*
    Created by Netlab 22214414
    Last updated on 2022/9/22
*/

#include "Streamplayer.h"

// 根据视频源创建AVFormatContext
AVFormatContext* Streamplayer:: create_de_formatc( const std::string& inputsource ) {

    AVFormatContext* avformatc = avformat_alloc_context();
    if ( !avformatc ) {
        throw std::runtime_error( "failed to alloc memory for avformat" );
    }

    AVDictionary* opts = nullptr;
    av_dict_set( &opts, "rtsp_transport", "tcp", 0);

    //open video
    int ret = avformat_open_input( &avformatc, inputsource.c_str(), nullptr, &opts );
    if ( ret != 0 ) {
        throw std::runtime_error( "failed to open input file" );
    }

    //find the input stream
    // it will be blocked if broadcaster doesn't send the stream
    std::cout << " waiting for the broadcast stream " << std::endl;
    ret = avformat_find_stream_info( avformatc, nullptr );
    if ( ret != 0 ) {
        throw std::runtime_error("failed to get stream info");
    }

    return avformatc;
}

// 创建视频源解码器相关信息
AVCodecContext* Streamplayer:: create_de_codecc( AVFormatContext* avformatc ) {

    assert( avformatc != nullptr );
    AVStream* de_stream = nullptr;
    // find stream index
    for ( int i = 0; i < avformatc->nb_streams; i++ ) {
        if ( avformatc->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO ) {
            this->video_index = i;
            de_stream = avformatc->streams[i];
            break;
        }
    }
    // find de_codec by codec_id
    AVCodec* de_codec = avcodec_find_decoder( de_stream->codecpar->codec_id );
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

    //use thread to speed up decode
    av_codecc->thread_count = 16;
    // open the de_codec using de_codec_context;
    ret = avcodec_open2( av_codecc, de_codec, nullptr);
    if ( ret < 0 ) {
        throw std::runtime_error("failed to open de_codec");
    }
    return av_codecc;

}

// 创建并设置编码器参数
void Streamplayer:: create_en_params( const std::string& video_output_path ) {

    //alloc memory for en stream
    avformat_alloc_output_context2(& this->en_formatc, nullptr, nullptr, video_output_path.c_str() );
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
    Streamplayer:: set_en_codecc( this->en_codecc );

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
void Streamplayer:: set_en_codecc( AVCodecContext* av_codecc ) {
    av_opt_set( av_codecc->priv_data, "preset", "ultrafast", 0);

    av_codecc->height = this->options.height;
    av_codecc->width = this->options.width;

    av_codecc->pix_fmt = AV_PIX_FMT_YUV420P;

    av_codecc->time_base = {1, TIME_BASE};
    av_codecc->framerate = {FPS, 1};

    av_codecc->bit_rate = 1000 * this->options.bit_rate ; 

    av_codecc->thread_count = 16;
}

// 初始化与保存视频相关的参数
void Streamplayer:: init_write_head( AVFormatContext* av_formatc, const std::string& video_output_path ) {
    
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
    //av_dict_set( &opts, "rtsp_transport", "tcp", 0 );

    // write header
    int ret = avformat_write_header( av_formatc, &opts );
    if ( ret < 0 ) {
        throw std::runtime_error("an error occurred when opening output file");
    }

}

/*
    接收函数，接收从RTSP读入的视频流，将相应的视频包信息传入接收队列中
*/
void Streamplayer:: receiving() {

    bool stop = false;
    AVPacket* input_packet = av_packet_alloc();
    int packet_num = 0;
    auto time1 = Utils::clock();
    while (!stop) {
        int ret = av_read_frame( this->de_formatc, input_packet );
        if ( ret < 0 ) {
            std::cout << "receiving ret < 0!" << std::endl;
            stop = true;
        } else{
            // skip audio stream, just process video stream
            if ( input_packet->stream_index != this->video_index ) {
                continue;
            }
            // push received packet into packet_queue
            this->received_packet_queue->push( av_packet_clone(input_packet) );
            packet_num++;
        }
        av_packet_unref( input_packet );
    }

    av_packet_free( &input_packet );
    auto time2 = Utils::clock();
    this->is_receive_done = true;
    std::cout << "Receiving done in : "<< Utils::get_duration(time1, time2)<<"ms, with numbers: "<< packet_num << std::endl;
}

/*
    解码读取到的视频流信息，得到低清视频帧
*/
void Streamplayer:: decoding() {

    auto time1 = Utils::clock();
    int frame_decoded_count = 0;
    // stop while when receiving is done and received_packet_queue is empty
    while ( !( this->is_receive_done && this->received_packet_queue->empty() ) ) {

        // waiting for the received_packet_queue
        if ( this->received_packet_queue->empty() ) {
            continue;
        }

        // get the received video packet
        AVPacket* received_packet = nullptr;
        this->received_packet_queue->pop( received_packet );

        // decode packet to get frame
        int ret = Streamplayer::decode_packet_to_queue( received_packet, frame_decoded_count );
        if ( ret < 0 ) {
            return;
        }

        av_packet_free( &received_packet );
    }

    // flush decoder
    int ret = Streamplayer::decode_packet_to_queue( nullptr, frame_decoded_count );
    if ( ret < 0 ) {
        return;
    }
    auto time2 = Utils::clock();
    this->is_decode_done = true;
    std::cout << "Decoding done: "<< Utils::get_duration(time1, time2)<<"ms, with numbers: "<< frame_decoded_count << std::endl;

}

// 解码得到低清的视频帧并传入到解码帧队列中
int Streamplayer:: decode_packet_to_queue(AVPacket* received_packet, int& frame_decoded_count) {

    AVFrame* input_frame = av_frame_alloc();

    int ret = avcodec_send_packet( this->de_codecc, received_packet );
    if (ret < 0) {
        std::cout << "Error while sending packet to decoder" << std::endl;
        return -1;
    }

    while ( ret >= 0 ) {
        // read decoded frame
        ret = avcodec_receive_frame( this->de_codecc, input_frame );
        if ( ret == AVERROR(EAGAIN) || ret == AVERROR_EOF ) {
            break;
        }else if (ret < 0) {
            std::cout<< "Error while receiving frame from decoder" << std::endl;
            return ret;
        }

        // push decoded frame to queue
        this->decoded_frame_queue->push( av_frame_clone(input_frame) );
        av_frame_unref( input_frame );
        frame_decoded_count++;

        if ( frame_decoded_count % this->options.sample_interval == 0  ) {
            //http_get_model();
            std::cout <<"frame decoded count:" << frame_decoded_count << std::endl;
        }
    }
    av_frame_free( &input_frame );

    return 0;
}

/*
    SR模型推理的主入口：
        1. 开启推理，合并的线程
        2. 主入口负责将低清的1080p视频帧田字裁剪成四部分，每一部分传入到相应的队列中
            推理线程开启四个子线程，每个线程使用一张GPU来进行SR推理
            合并线程将四个子线程得到SR推理后的结果合并成一个4K的视频帧
        3. 此外主入口还进行SR模型更新的操作，每当从推流端接收到新模型就会进行相应的更新
*/
void Streamplayer:: upsampling_main() {
    int frame_received_count = 0;
    
    std::thread upsampling_inference_thread(&Streamplayer::upsampling_inference, this);
    std::thread upsampling_merge_to_mat_thread(&Streamplayer::upsampling_merge_to_mat, this);
    auto time1 = Utils::clock();
    // stop while when decoding done and decoded_frame_queue is empty
    while ( !( this->is_decode_done && this->decoded_frame_queue->empty() ) ) {
        
        // waiting for the decoded_frame_queue
        if ( this->decoded_frame_queue->empty() ) {
            continue;
        }

        AVFrame* frame = nullptr;

        //get decoded frame
        this->decoded_frame_queue->pop( frame );

        std::shared_ptr<cv::Mat> input_mat_ptr = std::make_shared<cv::Mat>( Streamplayer:: av_frame_yuv420p_to_cv_mat_rgb(frame) );

        av_frame_free( &frame );

        //这里可以保存tmp得到输入的低清帧
        if ( this->is_test_psnr && frame_received_count % this->options.FPS == 0 ) {
            cv::Mat tmp = (*input_mat_ptr).clone();
            this->lr_mat_vec.push_back( tmp );
        }

        int crop_w = input_mat_ptr->cols / 2;
        int crop_h = input_mat_ptr->rows / 2;

        for ( int gpu_i = 0; gpu_i < this->gpu_count; gpu_i++ ) {
            int crop_x = gpu_i % 2;
            int crop_y = gpu_i / 2;
            
            cv::Mat crop_mat = (*input_mat_ptr)( cv::Rect(crop_x * crop_w, crop_y * crop_h, crop_w, crop_h) );

            this->lr_mat_queue_gpu[gpu_i]->push( std::make_pair( nullptr, std::make_shared<cv::Mat>( crop_mat.clone() ) ) );
        }
        
        frame_received_count++;

        // when model_queue_gpu is not empty update model
        if( Utils::is_none_empty( this->model_queue_gpu) ) {
            // update model
            std::cout << "update model! " << frame_received_count << std::endl;
            for ( int gpu_i = 0; gpu_i < this->gpu_count; gpu_i++ ) {
                this->model_queue_gpu[gpu_i]->consume_one(
                    [this, gpu_i]( torch::jit::script::Module& model ) {
                        std::lock_guard< std::mutex > lock( *(this->mtx_gpu[ gpu_i ]) );
                        this->models[gpu_i] = model;
                    }
                );
            }
        }

    }
    auto time2 = Utils::clock();
    this->is_upsample_main_done = true;
    std::cout << "upsampling main done " << Utils::get_duration(time1, time2)<<"ms, with numbers: " << frame_received_count << std::endl;

    upsampling_inference_thread.join();
    upsampling_merge_to_mat_thread.join();

    this->is_upsample_done = true;
    std::cout << "Upsampling done!" << std::endl;
}

/*
    推理线程开启四个子线程，每个线程使用一张GPU来进行SR推理
*/
void Streamplayer:: upsampling_inference() {   

    std::vector<std::thread> inference_thread( this->gpu_count );
    auto time1 = Utils::clock();
    for( int gpu_i = 0; gpu_i < this->gpu_count; gpu_i++ ) {

        inference_thread[gpu_i] = std::thread(
            [ gpu_i, this ]() {
                // stop while when upsample main is done and lr_mat_queue is empty
                while( !(this->is_upsample_main_done && this->lr_mat_queue_gpu[gpu_i]->empty() ) ){
                    // wait for the lr_mat_queue
                    if ( this->lr_mat_queue_gpu[gpu_i]->empty() ) {
                        continue;
                    }
                    std::pair<std::shared_ptr<torch::Tensor>, std::shared_ptr<cv::Mat>> crop_mat;
                    this->lr_mat_queue_gpu[gpu_i]->pop( crop_mat );

                    torch::jit::script::Module model;
                    // copy model to avoid conflit
                    {
                        std::lock_guard<std::mutex> lock( *(this->mtx_gpu[ gpu_i ]) );
                        model = this->models[gpu_i];
                    }
                    auto  sr_tensor_crop = std::make_shared<torch::Tensor>(
                         Streamplayer::inference( model, *(crop_mat.second), Streamplayer::get_cuda_num(gpu_i) )
                    );
                    this->sr_tensor_queue_gpu[gpu_i]->push( std::make_pair( sr_tensor_crop, nullptr) );
                }
            }
        );
    }
    for ( auto& th : inference_thread )  th.join();
    auto time2 = Utils::clock();

    this->is_upsample_infer_done = true;
    std::cout << "upsample infer done "<< Utils::get_duration(time1, time2)<<"ms, with numbers: " << std::endl;
}

/*
    合并线程将四个子线程得到SR推理后的结果合并成一个4K的视频帧
*/
void Streamplayer:: upsampling_merge_to_mat() {

    std::vector<std::thread> merge_thread( this->gpu_count );
    cv::Mat mat_buffer = cv::Mat::zeros( cv::Size(this->options.width, this->options.height) , CV_8UC3 );
    int merge_mat_number = 0;
    auto time1 = Utils::clock();
    while ( !(this->is_upsample_infer_done && Utils::is_all_empty(this->sr_tensor_queue_gpu) ) ) {

        if ( !Utils::is_none_empty(this->sr_tensor_queue_gpu) ) {
            continue;
        }

        for ( int gpu_i = 0; gpu_i < this->gpu_count; gpu_i++ ) {
            std::shared_ptr<TensorMatQueue>& sr_tensor_queue = this->sr_tensor_queue_gpu[gpu_i];

            merge_thread[ gpu_i ] = std::thread(
                [ gpu_i, &sr_tensor_queue, &mat_buffer ](){

                    int crop_x = gpu_i % 2;
                    int crop_y = gpu_i / 2;
                    std::pair<std::shared_ptr<torch::Tensor>, std::shared_ptr<cv::Mat>> crop_tensor;
                    
                    sr_tensor_queue->pop( crop_tensor );

                    cv::Mat crop_mat = Streamplayer::tensor_rgb_255_hwc_to_cv_mat_bgr( *(crop_tensor.first) );

                    crop_mat.copyTo( mat_buffer( cv::Rect( crop_x * crop_mat.cols, crop_y * crop_mat.rows,
                                             crop_mat.cols, crop_mat.rows) ) );
                }
            );
        }
        for ( auto& th : merge_thread ) th.join();
        
        if ( this->is_show ){
            this->playing_buffer_queue->push( std::make_pair( nullptr, std::make_shared<cv::Mat>( mat_buffer.clone() ) ) );
        }
        if ( this->is_save_video ) {
            this->encoded_mat_queue->push( std::make_pair( nullptr, std::make_shared<cv::Mat>( mat_buffer.clone() ) ) );
        }
        merge_mat_number++;
    }
    auto time2 = Utils::clock();
    std::cout << "upsampling merge to mat done "<< Utils::get_duration(time1, time2)<<"ms, with numbers: " << merge_mat_number << std::endl;
}

/*
    播放端将得到的4K帧进行播放
*/
void Streamplayer:: playing() {

    auto time1 = Utils::clock();
    int play_num = 0;
    // stop while when upsampling is done and playing_buffer_queue is empty
    while ( !( this->is_upsample_done && this->playing_buffer_queue->empty() ) ) {

        
        // waiting for the playing_buffer_queue
        if ( this->playing_buffer_queue->empty() ) {
            continue;
        }

        std::pair< std::shared_ptr<torch::Tensor>, std::shared_ptr<cv::Mat> > mat_show;

        this->playing_buffer_queue->pop( mat_show );

        //这里可以保存tmp得到SR帧
        if ( this->is_test_psnr && play_num % this->options.FPS == 0 ) {
            cv::Mat tmp = (*(mat_show.second)).clone();
            this->sr_mat_vec.push_back( tmp );
        }

        
        cv::namedWindow("Streamplayer", cv::WINDOW_NORMAL);
        cv::imshow("Streamplayer", *(mat_show.second) );
        cv::waitKey(1);

        play_num++;
    }
    auto time2 = Utils::clock();
    
    this->is_play_done = true;
    std::cout << "Playing done "<< Utils::get_duration(time1, time2)<<"ms, with numbers: " << play_num << std::endl;
}

/*
    播放端的起始函数：
        主要功能：1.从RTSP服务器拉流并解码得到低清视频帧 2.对低清视频帧进行SR推理得到4K视频帧
                    3. 根据需要选择直接播放4K视频帧还是再次编码保存视频
*/
void Streamplayer:: start() {

    std::cout << "Streamplayer start!" << std::endl;

    std::thread receiving_thread(&Streamplayer::receiving, this);

    std::thread decoding_thread(&Streamplayer::decoding, this);

    std::thread upsampling_main_thread(&Streamplayer::upsampling_main, this);

    std::thread http_serving_thread(&Streamplayer::http_serving, this);

    std::thread playing_thread;
    if ( this->is_show ) {
        playing_thread = std::thread(&Streamplayer::playing, this);
    }

    std::thread encoding_thread;
    if ( this->is_save_video ) {
        encoding_thread = std::thread(&Streamplayer::encoding, this);
    }
    // waiting for the thread
    http_serving_thread.detach();

    receiving_thread.join();

    decoding_thread.join();

    upsampling_main_thread.join();
    
    if ( this->is_show ) {
        playing_thread.join();
    }
    if ( this->is_save_video ) {
        encoding_thread.join();
    }
    if ( this->is_show ) {
        assert( this->is_play_done == true );
    }
    if ( this->is_save_video ) {
        assert( this->is_encode_done == true );
    }

    std::cout << "Streamplayer done!" << std::endl;
    
    if ( this->is_test_psnr ) {
        //可以添加测量psnr指标的部分，SR后的帧和lr帧都保存了，原始帧在推流端保存了
        // std::vector<double> psnr_vec;
        // int idx = 1;
        // double psnr = 0.0;
        // double psnr_sum = 0.0;
        // //std::cout << this->sr_mat_vec.size() << std::endl;
        // std::string ori_img_path = "/home/ubuntu/data/main/ffmpeg/frames/video_final/video_final_";
        // for ( auto& sr_img : this->sr_mat_vec ) {
        //     cv::Mat ori_img = cv::imread( ori_img_path + std::to_string(idx) + ".png" );

        //     psnr = Streamplayer::PSNR( ori_img, sr_img );

        //     psnr_vec.emplace_back( psnr );
        //     std::cout << "idx = " << idx << " psnr = " << psnr << std::endl;
        //     idx++;
        // }
        // for ( auto psnr : psnr_vec ) {
        //     psnr_sum += psnr;
        // }
        // std::cout << "Average psnr = " << psnr_sum / psnr_vec.size() << std::endl;
    }
}

// 保存SR后的视频时，视频编码部分
void Streamplayer:: encoding() {

    int frame_encoded_count = 0;
    auto time1 = Utils::clock();
    while( !(this->is_upsample_done && this->encoded_mat_queue->empty() ) ) {
        
        if ( this->encoded_mat_queue->empty() ) {
            continue;
        }

        std::pair< std::shared_ptr<torch::Tensor>, std::shared_ptr<cv::Mat> > mat_show;

        this->encoded_mat_queue->pop( mat_show );

        AVFrame* frame = nullptr;

        frame = Streamplayer::cv_mat_bgr_to_av_frame_yuv420p( *(mat_show.second) , frame_encoded_count );

        int ret = Streamplayer:: encode_avframe_to_stream( frame );
        if ( ret < 0 ) {
            std::cout << "encoding frames to video streaming fails!" << std::endl;
        }
        frame_encoded_count++;

        // since we don't use ffmpeg fuctions to alloc memory, just using av_frame_free can't free the memory
        // we use memcpy to alloc memory, so we have to use av_freep to free the memory
        av_freep(&frame->data[0]);
        av_frame_free(&frame);
    }
    //flush encoder to output the rest of frames
    int ret = Streamplayer:: encode_avframe_to_stream( nullptr );
    if ( ret < 0 ) {
        std::cout << "flushing the encode fails!" << std::endl;
    }
    av_write_trailer( this->en_formatc );

    auto time2 = Utils::clock();

    this->is_encode_done = true;

    std::cout << "Encoding done "<< Utils::get_duration(time1, time2)<<"ms, with numbers: " << frame_encoded_count << std::endl;
}

// 将编码后的视频写入文件中保存
int Streamplayer:: encode_avframe_to_stream( AVFrame* frame ) {

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
        output_packet->stream_index = this->video_index;
        output_packet->duration = this->en_stream->time_base.den / this->en_stream->time_base.num / FPS;
        output_packet->pts = av_rescale_q_rnd( output_packet->pts, {1, TIME_BASE}, this->en_stream->time_base, 
                                        (AVRounding)(AV_ROUND_NEAR_INF | AV_ROUND_PASS_MINMAX) );
        output_packet->dts = av_rescale_q_rnd( output_packet->dts, {1, TIME_BASE}, this->en_stream->time_base, 
                                        (AVRounding)(AV_ROUND_NEAR_INF | AV_ROUND_PASS_MINMAX) );

        // write packet 
        int ret = av_interleaved_write_frame( this->en_formatc, output_packet );
        if ( ret != 0 ) {
            std::cout << "Error while sending packet!" << std::endl;
            av_packet_free(&output_packet);
            return ret;
        }

        av_packet_unref( output_packet );
    }
    av_packet_free(&output_packet);

    return 0;
}

// SR推理部分
torch::Tensor Streamplayer:: inference(torch::jit::script::Module& model, cv::Mat& mat_frame, std::string device) {
    torch::NoGradGuard no_grad;
    model.eval();

    torch::Tensor input_tensor = torch::from_blob( mat_frame.data, 
                            {1, mat_frame.rows, mat_frame.cols, mat_frame.channels()}, torch::kUInt8 );

    input_tensor = input_tensor.to( torch::Device(device) );

    // GPU task
    // cv2 (b, h, w, c) RGB -> (b, c, h, w) RGB -> (b, c, h, w) RGB [0,1] tensor
    input_tensor = input_tensor.permute( {0, 3, 1, 2} ).contiguous().toType( torch::kFloat32 ).div(255.0);
    
    torch::Tensor output = model.forward( {input_tensor} ).toTensor();
    
    // (b, c, h, w) -> (c, h, w) -> (h, w, c)
    output = output.squeeze().detach().permute( {1, 2, 0} );
    output = output.mul(255).clamp(0, 255).to(torch::kUInt8);

    output = output.to(torch::kCPU);

    return output;
}

// http服务器 POST请求，当从推流端POST提交一个模型时，这里接收到就会更新
void Streamplayer:: http_serving() {
    httplib::Server server;
    std::string model_num;

    server.Post("/model",

        [&]( const httplib::Request& req, httplib::Response& res ) {
            model_num = req.get_file_value("Ack").content;
            std::cout << "http get model num: " << model_num << std::endl;

            std::istringstream model_str( req.get_file_value("model").content );
            for ( int gpu_i = 0; gpu_i < this->gpu_count; gpu_i++ ) {
                auto model_ = torch::jit::load( model_str, torch::Device( Streamplayer::get_cuda_num(gpu_i) ) );
                this->model_queue_gpu[ gpu_i ]->push( model_ );
            }
            
        }
    );

    server.listen("0.0.0.0", 9090);
}

// 将av_frame格式的视频帧转为cv Mat rgb格式
cv::Mat Streamplayer:: av_frame_yuv420p_to_cv_mat_rgb(AVFrame* frame) {
    int width = frame->width;
    int height = frame->height;
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
    cvtColor(img, rgb, cv::COLOR_YUV2RGB_I420);
    return rgb;
}

// 将cv Mat bgr格式的图片转为 av_frame yuv420p的视频帧
AVFrame* Streamplayer:: cv_mat_bgr_to_av_frame_yuv420p( cv::Mat& mat , int frame_number ) {

    AVFrame* frame = av_frame_alloc();
    frame->format = AV_PIX_FMT_YUV420P;
    frame->width = mat.cols;
    frame->height = mat.rows;
    frame->pts = (int64_t) ( frame_number *  TIME_BASE / FPS  );

    cv::Mat mat_yuv;
    cvtColor( mat, mat_yuv, cv::COLOR_BGR2YUV_I420 );

    av_image_alloc(frame->data, frame->linesize, frame->width, frame->height, 
                        static_cast<AVPixelFormat>(frame->format), 16);

    
    // ergodic rows
    int width = frame->width;
    int height = frame->height;
    std::vector<std::thread> convert_thread( 3 );

    // Y channel
    convert_thread[0] = std::thread(
        [&frame, &mat_yuv, width, height](){
            for ( int j = 0; j < height; j++ ) {
                memcpy( frame->data[0] + j * frame->linesize[0], mat_yuv.data + j * width, width);
            }
        }
    );
    
    // U channel
    convert_thread[1] = std::thread(
        [&frame, &mat_yuv, width, height](){
            for ( int j = 0; j < height / 2; j++ ) {
                memcpy( frame->data[1] + j * frame->linesize[1], 
                    mat_yuv.data + height * width + j * width / 2 , width / 2 );
            }
        }
    );

    // V channel
    convert_thread[2] = std::thread(
        [&frame, &mat_yuv, width, height](){
            for ( int j = 0; j < height / 2; j++ ) {
                memcpy(  frame->data[2] + j * frame->linesize[2], 
                        mat_yuv.data + height * width * 5 / 4 + j * width / 2, width / 2 );
            }
        }
    );
    
    for (auto& th: convert_thread ) th.join();

    return frame;
}

// 将tensor 类型的数据转为cv Mat bgr格式
cv::Mat Streamplayer:: tensor_rgb_255_hwc_to_cv_mat_bgr(torch::Tensor& tensor) {
    std::vector<cv::Mat> mat_channels;

    int height = tensor.size(0);
    int width = tensor.size(1);

    for ( int i = 0; i < 3; i++ ) {
        auto p = tensor.data_ptr<uint8_t>() + i * height * width;
        mat_channels.emplace_back( height, width, CV_8U, p);
    }
    cv::Mat mat;
    cv::merge( mat_channels, mat);
    cvtColor( mat, mat, cv::COLOR_RGB2BGR );

    return mat;
}

// 计算PSNR指标
double Streamplayer:: PSNR( cv::Mat m1,  cv::Mat m2) {
    m1.convertTo( m1, CV_32F );
    m2.convertTo( m2, CV_32F );
    cv::Mat differ( m1.size(), CV_32F );

    cv::absdiff( m1, m2, differ );      // | m1 - m2|
    differ = differ.mul( differ );        // | m1 - m2|^2
    cv::Scalar s = cv::sum( differ );   
    double N = m1.channels() * m1.total();

    double sse;
    if ( m1.channels() == 3 ){
        sse = s.val[0] + s.val[1] + s.val[2];
    }else {
        sse = s.val[0];
    }

    if ( std::isnan(sse) ) {
        return 0;
    }else {
        double mse = sse / N;
        double psnr = 10.0 * std::log10( (255 * 255) / (mse + 1e-6) );
        return psnr;
    }
}

// 返回相应的推理用的cuda编号
std::string Streamplayer:: get_cuda_num( int idx ) {
    return  "cuda:" + std::to_string( this->options.cuda_num[idx] ) ;
}