/*
    在线训练数据集设置
    将push进入的低清输入和原始高清结果转换为torch::Tensor格式
    并进行相应的格式变换(h,c,w) -> (n,c,patch_size,patch_size)
*/
#ifndef __BRODATASET__
#define __BRODATASET__

#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include <chrono>
#include <utility>
#include <string>
using namespace torch::indexing;

class PatchImgDataset : public torch::data::Dataset<PatchImgDataset> {
    public:
        std::vector< std::pair<torch::Tensor, torch::Tensor> > img_queue = {};
        //std::string device;
        size_t patch_size;
        size_t repeat;
        size_t scale;
        std::string device;

    public:
        PatchImgDataset(size_t patch_size = 48, size_t repeat = 120, size_t scale = 2, std::string device = "cuda:0");

        torch::data::Example<> get(size_t index) override ;

        torch::optional<size_t> size() const override ;

        void push( const cv::Mat& lr, const cv::Mat& hr );

        void clear();

        //utility
        static torch::Tensor blocking( const torch::Tensor &t, int patch_size, int block_num_h, int block_num_w );

        static torch::nn::functional::PadFuncOptions get_padding_options(int right, int bottom);
};

// ********  Achieve PatchImgDataset: ********  //

inline PatchImgDataset:: PatchImgDataset(size_t patch_size, size_t repeat, size_t scale, std::string device) {
    this->patch_size = patch_size;
    this->repeat = repeat;
    this->scale = scale;
    this->device = device;
}

inline void PatchImgDataset:: clear() {
    this->img_queue.clear();
}

inline torch::optional<size_t> PatchImgDataset:: size() const {
    return this->repeat * this->img_queue.size();
}

inline torch::data::Example<> PatchImgDataset:: get( size_t index ) {

    std::pair<torch::Tensor, torch::Tensor> img_data = this->img_queue[ index % (this->img_queue).size() ];

    torch::Tensor& img_lr_patches = img_data.first;
    torch::Tensor& img_hr_patches = img_data.second;
    
    int num_patches = img_lr_patches.size(0);

    int idx = torch::randint( 0, num_patches, {1}).item().toInt();

    return { img_lr_patches.index({idx}), img_hr_patches.index({idx}) };

}

inline void PatchImgDataset:: push( const cv::Mat& lr, const cv::Mat& hr ) {
    
    int lr_w = lr.size().width, lr_h = lr.size().height;
    int hr_w = hr.size().width, hr_h = hr.size().height;
    
    int padding_pixel_number_right = (this->patch_size - (lr_w % this->patch_size)) % this->patch_size;
    int padding_pixel_number_bottom = (this->patch_size - (lr_h % this->patch_size)) % this->patch_size;

    torch::Tensor lr_tensor = torch::from_blob(lr.data , {lr_h, lr_w, lr.channels()} , torch::kUInt8);
    lr_tensor = lr_tensor.to( torch::Device(this->device) );

    torch::Tensor hr_tensor = torch::from_blob(hr.data, {hr_h, hr_w, hr.channels()} , torch::kUInt8 );
    hr_tensor = hr_tensor.to( torch::Device(this->device) );
    
    //padding zeros
    lr_tensor = torch::nn::functional::pad(lr_tensor, PatchImgDataset::get_padding_options(padding_pixel_number_right,
                                                                              padding_pixel_number_bottom));
    hr_tensor = torch::nn::functional::pad(hr_tensor, PatchImgDataset::get_padding_options(padding_pixel_number_right * scale,
                                                                              padding_pixel_number_bottom * scale));

    // cv2 (h,w,c) BGR -> (c,h,w) BGR -> (c,h,w) RGB
    lr_tensor = lr_tensor.permute( {2, 0, 1} ).flip( {0} );
    hr_tensor = hr_tensor.permute( {2, 0, 1} ).flip( {0} );;

    // blocking  (c,h,w) -> (n, c, h, w) [0,1]
    int block_num_h = int( lr_tensor.size(1) ) / this->patch_size;
    int block_num_w = int( lr_tensor.size(2) ) / this->patch_size;

    torch::Tensor lr_patches = PatchImgDataset::blocking(lr_tensor, this->patch_size, block_num_h, block_num_w);
    torch::Tensor hr_patches = PatchImgDataset::blocking(hr_tensor, this->patch_size * this->scale, block_num_h, block_num_w);

    // lr_patches:(n, c, 48, 48) ; hr_patches:(n, c, 96, 96)
    this->img_queue.emplace_back( std::make_pair(lr_patches, hr_patches) );
}


inline torch::Tensor PatchImgDataset:: blocking( const torch::Tensor &t, int patch_size, int block_num_h, int block_num_w ) {
    // (c,h,w) [0,255] -> (n, c, h, w) [0,1]
    return t.reshape( {3, block_num_h, patch_size, block_num_w, patch_size} )
            .transpose(2, 3).reshape( {3, block_num_h*block_num_w, patch_size, patch_size})
            .permute({1, 0, 2, 3}).contiguous().to( torch::kFloat32 ).div(255.0);

}

//填充0
inline torch::nn::functional::PadFuncOptions PatchImgDataset:: get_padding_options(int right, int bottom) {
    auto pad_options = torch::nn::functional::PadFuncOptions({0, 0, 0, right, 0, bottom});
    pad_options.mode(torch::kConstant);
    pad_options.value(0);
    return pad_options;
}

#endif // __BRODATASET__