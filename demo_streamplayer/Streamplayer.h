#ifndef __STREAMPLAYER__
#define __STREAMPLAYER__

#include <iostream>
#include <vector>
#include "Setting.h"
#include <thread>
#include "boost/lockfree/spsc_queue.hpp"
#include <filesystem>
#include <cassert>
#include <chrono>
//#include "fmt/core.h"
#include "Utils.h"
#include <torch/script.h>
#include <opencv2/opencv.hpp>
#include <httplib.h>
#include <mutex>
#include <ctime>
// ********************  FFmpeg includes *******************
#ifdef __cplusplus
    extern "C"{
#endif

    #include <libavcodec/avcodec.h>
    #include <libavformat/avformat.h>
    #include <libavutil/time.h>
    #include <libavutil/opt.h>
    #include "libavutil/imgutils.h"
    #include <libswscale/swscale.h>

#ifdef __cplusplus
    }
#endif

// boost::lockfree::spsc_queue<>, 单生产者，单消费者的无锁队列
using AVFrameQueue = boost::lockfree::spsc_queue< AVFrame *, boost::lockfree::capacity<4096> >;
using AVPacketQueue = boost::lockfree::spsc_queue<AVPacket *, boost::lockfree::capacity<4096>>;
using TensorMatQueue = boost::lockfree::spsc_queue< std::pair<std::shared_ptr<torch::Tensor>, std::shared_ptr<cv::Mat>>, boost::lockfree::capacity<40960> >;
using ModelQueue = boost::lockfree::spsc_queue< torch::jit::script::Module, boost::lockfree::capacity<40960> >;

const int64_t TIME_BASE = 6 * 1000 * 1000 ;
const int64_t FPS = 30;

class Streamplayer {
    public:
        std::string video_rtsp_address;
        int video_index;
        int gpu_count;
        //decoder
        AVFormatContext* de_formatc;
        AVCodecContext* de_codecc;

        //encoder
        AVFormatContext* en_formatc;
        AVCodecContext* en_codecc;
        AVStream* en_stream;

        //status
        bool is_receive_done = false;
        bool is_decode_done = false;
        bool is_upsample_done = false;
        bool is_upsample_main_done = false;
        bool is_upsample_infer_done = false;
        //
        bool is_play_done = false;
        bool is_encode_done = false;
        //evaluate
        bool is_test_psnr = true;
        bool is_show = false;
        bool is_save_video = false;
        std::string video_output_path;
        //model
        std::vector< torch::jit::script::Module > models;

        //queue
        std::shared_ptr<AVPacketQueue> received_packet_queue;
        std::shared_ptr<AVFrameQueue> decoded_frame_queue;
        std::shared_ptr<TensorMatQueue> playing_buffer_queue;
        std::shared_ptr<TensorMatQueue> encoded_mat_queue;

        std::vector< std::shared_ptr<ModelQueue> > model_queue_gpu;
        std::vector< std::shared_ptr<TensorMatQueue> > lr_mat_queue_gpu;
        std::vector< std::shared_ptr<TensorMatQueue> > sr_tensor_queue_gpu;

        //evaluate
        std::vector< cv::Mat > sr_mat_vec;
        std::vector< cv::Mat > lr_mat_vec;
        //setting
        Setting options;

        //mutex to protect model since updating and inferencing may happen at one time
        //
        std::vector< std::shared_ptr<std::mutex> >  mtx_gpu;
        
    public:
        //init
        Streamplayer( const std::string& video_rtsp_address );
        Streamplayer( const Setting& options );
        ~Streamplayer();

        // create_de
        AVFormatContext* create_de_formatc( const std::string& inputsource );
        
        AVCodecContext* create_de_codecc( AVFormatContext* avformatc );

        // create_en
        void create_en_params( const std::string& video_output_path );

        void set_en_codecc( AVCodecContext* av_codecc ); 

        // create writter head
        void init_write_head( AVFormatContext* av_formatc, const std::string& video_output_path );

        // receiving
        void receiving();

        // decode
        int decode_packet_to_queue(AVPacket* received_packet, int& frame_decoded_count);

        void decoding();

        //upsample flow line achieved
        void upsampling_main();
        void upsampling_inference();
        void upsampling_merge_to_mat();

        //player
        void playing();

        //start
        void start();

        //encode
        void encoding();
        int encode_avframe_to_stream( AVFrame* frame );

        // model inference
        torch::Tensor inference(torch::jit::script::Module& model, cv::Mat& mat_frame, std::string device);
        
        // http Get model
        void http_get_model();
        void http_serving();
        
        //utility
        static cv::Mat av_frame_yuv420p_to_cv_mat_rgb(AVFrame* frame);

        static AVFrame* cv_mat_bgr_to_av_frame_yuv420p( cv::Mat& mat , int frame_number);

        static cv::Mat tensor_rgb_255_hwc_to_cv_mat_bgr(torch::Tensor& tensor);

        static double PSNR( cv::Mat m1,  cv::Mat m2);

        std::string get_cuda_num( int idx );
};

inline Streamplayer:: Streamplayer( const Setting& options ) {
    this->options = options;

    this->video_rtsp_address = this->options.rtsp_server;

    this->is_test_psnr = this->options.is_test_psnr;
    this->is_show = this->options.is_show;
    this->is_save_video = this->options.is_save_video;
    this->gpu_count = 4;

    // resize the queue
    this->lr_mat_queue_gpu.resize( this->gpu_count );
    this->sr_tensor_queue_gpu.resize( this->gpu_count );
    this->models.resize( this->gpu_count );
    this->model_queue_gpu.resize( this->gpu_count );
    this->mtx_gpu.resize( this->gpu_count );
    //init model list
    std::vector<std::thread> init_model_thread( this->gpu_count );
    for ( int i = 0; i < this->gpu_count; i++ ) {
        init_model_thread[i] = std::thread(
            [i, this](){
                this->models[i] = torch::jit::load( this->options.pretrain_model_path, 
                                                torch::Device( Streamplayer::get_cuda_num(i) ) );
            }
        );
    }
    for (auto & th: init_model_thread ) th.join();
    std::cout << "load pretrain model done!" << std::endl;

    //queue
    this->received_packet_queue = std::make_shared<AVPacketQueue>();
    this->decoded_frame_queue = std::make_shared<AVFrameQueue>();
    this->playing_buffer_queue = std::make_shared<TensorMatQueue>();
    this->encoded_mat_queue = std::make_shared<TensorMatQueue>();
    
    for ( int i = 0; i < this->gpu_count; i++ ) {
        this->model_queue_gpu[i] = std::make_shared<ModelQueue>();
        this->lr_mat_queue_gpu[i] = std::make_shared<TensorMatQueue>();
        this->sr_tensor_queue_gpu[i] = std::make_shared<TensorMatQueue>();
        this->mtx_gpu[i] = std::make_shared< std::mutex >();
    }

    // create encoder when not directly showing frames
    if ( this->is_save_video ) {
        this->video_output_path = this->options.video_output_path;
        Streamplayer::create_en_params( this->video_output_path );

        Streamplayer::init_write_head( this->en_formatc, this->video_output_path );
    }

    // create decoder
    // since it will wait for the broadcast stream, it should be placed in the last of init
    this->de_formatc = Streamplayer::create_de_formatc( this->video_rtsp_address );
    this->de_codecc = Streamplayer::create_de_codecc( this->de_formatc );

    std::cout << "Initialize Streamplayer done!" << std::endl;
}

inline Streamplayer:: ~Streamplayer() {

    avformat_close_input( &this->de_formatc );
    avformat_free_context( this->de_formatc );
    this->de_formatc = nullptr;
    avcodec_free_context( &this->de_codecc );
    this->de_codecc = nullptr;

    std::cout << "~Streamplayer done!" << std::endl;
}

#endif // __STREAMPLAYER__
// Achieved in STREAMPLAYER.CPP