/*
    读取option.toml配置文件
*/
#ifndef __SETTING__
#define __SETTING__

#include <toml.hpp>
#include <iostream>


class Setting{
    public:
        //[Url]
        std::string hr_video_input;
        std::string rtsp_server;
        std::string http_server;

        //[OnlineTraining]
        int sample_interval;
        int training_cycle;
        int sampling_cycle;

        std::vector<int> cuda_num;
        std::string pretrain_model_path;

        double init_lr;
        int lr_decay_epoch;
        double lr_decay_rate;
        double weight_decay;

        int training_epoch_num;
        int batch_size;
        int patch_size;
        int dataset_repeat;
        int scale;

        //[Video]
        int bit_rate;
        std::string preset;
    public:
        Setting(const std::string& option_path);
        Setting(){}

};

inline Setting::Setting( const std::string& option_path ) {
    auto option = toml::parse( option_path );

    //[Url]
    this->hr_video_input = toml::find<std::string>(option, "Url", "hr_video_input");
    this->rtsp_server = toml::find<std::string>(option, "Url", "rtsp_server");
    this->http_server = toml::find<std::string>(option, "Url", "http_server");

    //[OnlineTraining]
    this->sample_interval = toml::find<int>(option, "OnlineTraining", "sample_interval");
    this->training_cycle = toml::find<int>(option, "OnlineTraining", "training_cycle");
    this->sampling_cycle = toml::find<int>(option, "OnlineTraining", "sampling_cycle");

    this->cuda_num = toml::find<std::vector<int>>(option, "OnlineTraining", "cuda");
    this->pretrain_model_path = toml::find<std::string>(option, "OnlineTraining", "pretrain_model_path");

    this->init_lr = toml::find<double>(option, "OnlineTraining", "init_lr");
    this->lr_decay_rate = toml::find<double>(option, "OnlineTraining", "lr_decay_rate");
    this->lr_decay_epoch = toml::find<int>(option, "OnlineTraining", "lr_decay_epoch");
    this->weight_decay = toml::find<double>(option, "OnlineTraining", "weight_decay");

    this->training_epoch_num = toml::find<int>(option, "OnlineTraining", "training_epoch_num");
    this->batch_size = toml::find<int>(option, "OnlineTraining", "batch_size");
    this->patch_size = toml::find<int>(option, "OnlineTraining", "patch_size");
    this->dataset_repeat = toml::find<int>(option, "OnlineTraining", "dataset_repeat");
    this->scale = toml::find<int>(option, "OnlineTraining", "scale");
    //[Video]
    this->bit_rate = toml::find<int>(option, "Video", "bit_rate");
    this->preset = toml::find<std::string>(option, "Video", "preset");
}

#endif //SETTING.H


