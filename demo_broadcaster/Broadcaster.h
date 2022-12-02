#ifndef __BROADCASTER__
#define __BROADCASTER__

#include <iostream>
#include <tuple>
#include <map>
#include <utility>
#include <vector>
#include <cassert>
#include <string>
#include <filesystem>
#include "Setting.h"
#include <thread>
#include "boost/lockfree/spsc_queue.hpp"
#include <chrono>
#include "Utils.h"
#include "httplib.h"
#include "OnlineTraining.h"
#include <torch/script.h>
#include <opencv2/opencv.hpp>
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
using AVFrameQueue = boost::lockfree::spsc_queue< AVFrame* , boost::lockfree::capacity<4096> >;
using AVPacketQueue = boost::lockfree::spsc_queue< AVPacket* , boost::lockfree::capacity<4096> >;
using MatFrameNumberQueue = boost::lockfree::spsc_queue< std::pair<cv::Mat, int>, boost::lockfree::capacity<4096> >;
using ModelFrameNumberQueue = boost::lockfree::spsc_queue< std::pair<std::string, int> , boost::lockfree::capacity<4096> >;

const int64_t TIME_BASE = 6 * 1000 * 1000;
const int64_t FPS = 30;

class Broadcaster {

    public:

        std::string hr_video_input_path;
        std::string video_output_path;
        int hr_video_index;
        int gpu_count;

        //decoder 
        AVFormatContext* de_formatc_hr;
        AVCodecContext* de_codecc_hr;
        AVCodecContext* de_codecc_lr;

        //encoder
        AVFormatContext* en_formatc;
        AVCodecContext* en_codecc;
        AVStream* en_stream;

        //status
        bool is_decode_hr_done = false;
        bool is_decode_lr_done = false;

        bool is_encode_done = false;
        bool is_broadcast_done = false;

        // queue
        std::shared_ptr<AVFrameQueue> decoded_frame_queue;
        std::shared_ptr<AVPacketQueue> encoded_packet_queue;
        std::shared_ptr<AVPacketQueue> encoded_packet_for_dec_queue;
        std::vector< std::shared_ptr<MatFrameNumberQueue> > hr_mat_queue_list;
        std::vector< std::shared_ptr<MatFrameNumberQueue> > lr_mat_queue_list;

        std::shared_ptr<ModelFrameNumberQueue> model_queue;
        //online training
        std::shared_ptr<OnlineTraining> trainer;

        //setting
        Setting options;

    public:
        // init
        Broadcaster( const Setting& options );
        //deconstrcut
        ~Broadcaster();

        // create_de    
        AVFormatContext* create_de_formatc( const std::string& inputsource );     

        AVCodecContext* create_de_codecc( AVFormatContext* avformatc, int thread_num, int& video_index);  

        AVCodecContext* create_lr_de_codecc();
        // create_en    
        void create_en_params( const std::string& video_output_path );   

        void set_av_codecc( AVCodecContext* av_codecc );     

        // create writter head
        void init_write_head( AVFormatContext* av_formatc, const std::string& video_output_path );   

        // decoding 
        void decoding_hr();
        void decoding_lr();

        //encoding
        int encode_frame_to_queue(AVFrame* frame);   
        void scale_av_frame( AVFrame* frame, AVFrame* output_frame, int width, int height );    
        void encoding();

        // broadcasting
        void broadcasting();

        // main start
        void start();

        // online training
        void online_training( int frame_number );

        // utility 
        static cv::Mat av_frame_yuv420p_to_cv_mat_bgr(AVFrame* frame);
        std::string get_cuda_num( int idx );
};

inline Broadcaster:: Broadcaster( const Setting& options ) {

    this->options = options;
    this->gpu_count = this->options.cuda_num.size();

    this->hr_video_input_path = this->options.hr_video_input;
    this->video_output_path = this->options.rtsp_server;
    //decoder hr
    this->de_formatc_hr = Broadcaster::create_de_formatc(this->hr_video_input_path);
    this->de_codecc_hr = Broadcaster::create_de_codecc( this->de_formatc_hr, 16, this->hr_video_index);
    
    this->de_codecc_lr = Broadcaster::create_lr_de_codecc();
    //encoder
    Broadcaster::create_en_params( this->video_output_path );

    //init write head
    Broadcaster::init_write_head( this->en_formatc, this->video_output_path);

    //queue

    this->hr_mat_queue_list.resize( this->gpu_count );
    this->lr_mat_queue_list.resize( this->gpu_count );
    this->decoded_frame_queue = std::make_shared<AVFrameQueue>();
    this->encoded_packet_queue = std::make_shared<AVPacketQueue>();
    this->encoded_packet_for_dec_queue = std::make_shared<AVPacketQueue>();
    this->model_queue = std::make_shared<ModelFrameNumberQueue>();

    for ( int i = 0; i < this->gpu_count; i++ ) {
        this->hr_mat_queue_list[i] = std::make_shared<MatFrameNumberQueue>();
        this->lr_mat_queue_list[i] = std::make_shared<MatFrameNumberQueue>();;
    }

    //online training
    this->trainer = std::make_shared<OnlineTraining>( this->options );

    std::cout << "Initialize broadcaster done!" << std::endl;
}

inline Broadcaster:: ~Broadcaster() {

    avformat_close_input( &this->de_formatc_hr );
    avformat_free_context( this->de_formatc_hr );
    this->de_formatc_hr = nullptr;
    avcodec_free_context( &this->de_codecc_hr );
    this->de_codecc_hr = nullptr;

    avcodec_free_context( &this->de_codecc_lr );
    this->de_codecc_lr = nullptr;

    avformat_free_context( this->en_formatc );
    this->en_formatc = nullptr;
    avcodec_free_context( &this->en_codecc );
    this->en_codecc = nullptr;

    std::cout << "~Broadcaster done!" << std::endl;
}

#endif  // __BROADCASTER__ 
// Achieved in BROADCASTER.CPP