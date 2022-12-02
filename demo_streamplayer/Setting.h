#ifndef __SETTING__
#define __SETTING__

#include <toml.hpp>
#include <iostream>


class Setting{
    public:
        //[Url]
        std::string rtsp_server;
        std::string video_output_path;
        int http_port;
        //[OnlineTraining]
        int sample_interval;
        std::string pretrain_model_path;

        //[Player]
        bool is_test_psnr;
        bool is_show;
        bool is_save_video;
        std::vector<int> cuda_num;

        //[Video]
        int height;
        int width;
        int FPS;
        int bit_rate;
    public:
        Setting(const std::string& option_path);
        Setting(){}

};

inline Setting::Setting( const std::string& option_path ) {
    auto option = toml::parse( option_path );

    //[Url]
    this->rtsp_server = toml::find<std::string>(option, "Url", "rtsp_server");
    this->video_output_path = toml::find<std::string>(option, "Url", "video_output_path");
    this->http_port = toml::find<int>(option, "Url", "http_port");
    //[OnlineTraining]
    this->sample_interval = toml::find<int>(option, "OnlineTraining", "sample_interval");
    this->pretrain_model_path = toml::find<std::string>(option, "OnlineTraining", "pretrain_model_path");
    this->cuda_num = toml::find<std::vector<int>>(option, "OnlineTraining", "cuda");

    //[Player]
    this->is_test_psnr = toml::find<bool>(option, "Player", "is_test_psnr");
    this->is_show = toml::find<bool>(option, "Player", "is_show");
    this->is_save_video = toml::find<bool>(option, "Player", "is_save_video");

    //[Video]
    this->height = toml::find<int>(option, "Video", "height");
    this->width = toml::find<int>(option, "Video", "width");
    this->FPS = toml::find<int>(option, "Video", "FPS");
    this->bit_rate = toml::find<int>(option, "Video", "bit_rate");
}

#endif //SETTING.H


