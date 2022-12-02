/*
    多卡在线训练
*/

#ifndef __ONLINETRAINING__
#define __ONLINETRAINING__

#include <iostream>
#include <shared_mutex>
#include <torch/script.h>
#include <torch/torch.h>
#include <functional>
#include <chrono>
#include <utility>
#include "Utils.h"
#include "PatchImgDataset.h"
#include "Setting.h"
#include <opencv2/opencv.hpp>
#include <condition_variable>

using MatFrameNumberQueue = boost::lockfree::spsc_queue< std::pair<cv::Mat, int>, boost::lockfree::capacity<4096> >;
using GradientsQueue = boost::lockfree::spsc_queue< torch::Tensor, boost::lockfree::capacity<4096> >;

//路障，用于同步多个线程, 保证每个线程都到达此处后程序才会继续
class Barrier {
public:
     explicit Barrier(std::size_t iCount = 0) : 
      mThreshold(iCount), 
      mCount(iCount), 
      mGeneration(0) {
    }

    void set(std::size_t iCount) {
        this->mThreshold = iCount;
        this->mCount = iCount;
        this->mGeneration = 0;
    }

    void Wait() {
        std::unique_lock<std::mutex> lLock{mMutex};
        auto lGen = mGeneration;
        if (!--mCount) {
            mGeneration++;
            mCount = mThreshold;
            mCond.notify_all();
        } else {
            mCond.wait(lLock, [this, lGen] { return lGen != mGeneration; });
        }
    }

private:
    std::mutex mMutex;
    std::condition_variable mCond;
    std::size_t mThreshold;
    std::size_t mCount;
    std::size_t mGeneration;
};

class OnlineTraining {
    public:
        //training
        std::vector< std::shared_ptr< PatchImgDataset > > dataset_list;
        std::vector< std::vector<torch::Tensor> > model_parameters_list;
        std::vector< std::shared_ptr<torch::optim::Adam> > optimizer_list;
        std::vector< torch::jit::script::Module > models;
        int gpu_count;

        //Allreduce 
        std::vector< std::shared_ptr<GradientsQueue> > gradients_queue_list;
        Barrier barrier0;
        Barrier barrier1;
        Barrier barrier2;
        Barrier barrier_dataset;

        int total_epoch = 0;
        Setting options;
        
    public:
        OnlineTraining( const Setting& options );

        static torch::Tensor loss_func( const torch::Tensor& output, const torch::Tensor& target,
                                            int reduction = torch::Reduction::Mean);

        void adjust_learning_rate( int my_rank, int epoch ) ;

        template<typename Dataloader> bool train_one_epoch(int my_rank, Dataloader& dataloader);

        void Allreduce();

        bool training( std::vector< std::shared_ptr<MatFrameNumberQueue> > &lr_mat_queue_list,
                        std::vector< std::shared_ptr<MatFrameNumberQueue> > &hr_mat_queue_list );

        void training_threads( int my_rank, std::shared_ptr<MatFrameNumberQueue>& lr_mat_queue,
                                            std::shared_ptr<MatFrameNumberQueue>& hr_mat_queue );

        std::string get_model_data() ;

        std::string get_cuda_num( int idx );

};

// ************* Achieve OnlineTraining: ************** //

//构造函数，初始化相关参数
inline OnlineTraining:: OnlineTraining( const Setting& options ) {

    this->options = options;
    this->gpu_count = this->options.cuda_num.size();

    //resize the queue
    this->models.resize( this->gpu_count );
    this->model_parameters_list.resize( this->gpu_count );
    this->dataset_list.resize( this->gpu_count );
    this->optimizer_list.resize( this->gpu_count );
        //allreduce
    this->gradients_queue_list.resize( this->gpu_count );
    this->barrier0.set( this->gpu_count );
    this->barrier1.set( this->gpu_count );
    this->barrier2.set( this->gpu_count );
    this->barrier_dataset.set(this->gpu_count);
    //init model list    
    std::vector<std::thread> init_thread( this->gpu_count );
    for ( int i = 0; i < this->gpu_count; i++ ) {
        init_thread[i] = std::thread(
            [i, this]() {
                //init models
                this->models[i] = torch::jit::load( this->options.pretrain_model_path,
                                                torch::Device( OnlineTraining::get_cuda_num(i) ) );
                //get models params
                for ( const auto& params : this->models[i].parameters() ) {
                    this->model_parameters_list[i].push_back( params );
                }
                //init datset
                this->dataset_list[i] = std::make_shared< PatchImgDataset >(
                    this->options.patch_size,
                    this->options.dataset_repeat,
                    this->options.scale,
                    OnlineTraining::get_cuda_num(i) 
                );
                //init optimizer
                this->optimizer_list[i] = std::make_shared< torch::optim::Adam > ( 
                    torch::optim::Adam(
                        this->model_parameters_list[i], torch::optim::AdamOptions(this->options.init_lr).weight_decay( this->options.weight_decay )
                    ) 
                );

                this->gradients_queue_list[i] = std::make_shared<GradientsQueue>();
            }
        );
    }
    for ( auto& th: init_thread ) th.join();
    std::cout << "OnlineTraining init done!" << std::endl;
}

// 计算误差的函数，使用L1误差
inline torch::Tensor OnlineTraining:: loss_func( const torch::Tensor& output, const torch::Tensor& target, 
                                            int reduction ) {
    return torch::l1_loss( output, target, reduction );
}

// 调整学习率
inline void OnlineTraining:: adjust_learning_rate( int my_rank, int epoch ) {

    auto new_lr = this->options.init_lr * cv::pow( this->options.lr_decay_rate , epoch / this->options.lr_decay_epoch );

    for (auto param_group : this->optimizer_list[my_rank]->param_groups()) {
        //Static cast needed as options() returns OptimizerOptions (base class)
        static_cast<torch::optim::AdamOptions &>(param_group.options()).lr(new_lr);
    }

}

// 聚集所有线程的梯度并求和，将结果广播到每个线程
inline void OnlineTraining:: Allreduce() {

    std::vector< std::vector<torch::Tensor> > gradients_vec(this->gpu_count);
    std::vector< std::thread > allreduce_thread( this->gpu_count );

    for ( int gpu_i = 0; gpu_i < this->gpu_count; gpu_i++ ) {
        std::shared_ptr<GradientsQueue>& gradients_queue = this->gradients_queue_list[gpu_i];

        allreduce_thread[gpu_i] = std::thread(
            [gpu_i, &gradients_queue, &gradients_vec ]() {
                //stop while when all gpus pushing params to gradients_queue
                while( !gradients_queue->empty()  ) {
                    torch::Tensor tmp;
                    gradients_queue->pop( tmp );
                    gradients_vec[gpu_i].push_back( tmp );

                }
            }
        );

    }
    for( auto& th : allreduce_thread ) th.join();

    //reduce: sum all the gradients
    for ( int params_i = 0; params_i < gradients_vec[0].size(); params_i++ ) {
        for ( int gpu_i = 1; gpu_i < this->gpu_count; gpu_i++ ) {
            gradients_vec[0][ params_i ] += gradients_vec[gpu_i][params_i];
        }
    }

    //broadcaster all the gradients
    for ( int gpu_i = 0; gpu_i < this->gpu_count; gpu_i++ ) {
        std::shared_ptr<GradientsQueue>& gradients_queue = this->gradients_queue_list[gpu_i];
        assert( gradients_queue->empty() );
        allreduce_thread[gpu_i] = std::thread(
            [gpu_i, &gradients_queue, &gradients_vec ](){

                for ( int params_i = 0; params_i < gradients_vec[0].size(); params_i++ ) {
                    gradients_queue->push( gradients_vec[0][params_i] );
                }

            }
        );
    }
    for( auto& th : allreduce_thread ) th.join();
}

/*
    训练时，每个线程使用loss.backward()得到对应的梯度后
    在主线程聚集相应的梯度求和，再广播到每个线程，此操作即为Allreduce
    每个线程得到更新后的梯度即可进行相应的梯度下降
*/
template<typename Dataloader>
inline bool OnlineTraining:: train_one_epoch( int my_rank, Dataloader& dataloader ) {
    OnlineTraining::adjust_learning_rate( my_rank,  this->total_epoch );

    auto& model = this->models[my_rank];
    auto& optimizer = this->optimizer_list[my_rank];
    auto& model_parameters = this->model_parameters_list[my_rank];
    std::string device = OnlineTraining::get_cuda_num( my_rank );

    for( auto& batch : dataloader ) {

        auto img_lr = batch.data.to( device );
        auto img_hr = batch.target.to( device );
        auto img_sr = model.forward( {img_lr} ).toTensor();
        auto loss = OnlineTraining::loss_func( img_sr, img_hr );

        optimizer->zero_grad();
        loss.backward();
        
        for( const auto& param : model_parameters ) {
            this->gradients_queue_list[my_rank]->push( param.grad().data().to(torch::kCPU) );
        }
        this->barrier0.Wait();

        if ( my_rank == 0 ) {
            OnlineTraining::Allreduce();
        }
        // waiting for allreduce
        this->barrier1.Wait();

        torch::Tensor gradients_tmp;
        for ( int params_i = 0; params_i < model_parameters.size(); params_i++ ) {
            this->gradients_queue_list[my_rank]->pop( gradients_tmp );
            model_parameters[params_i].grad().data() = gradients_tmp.to(device) / this->gpu_count;
        }

        optimizer->step();

        //barrier: synchronize all threads
        this->barrier2.Wait();
    }

    if ( my_rank == 0 ) {
        this->total_epoch++;
    }
    return true;
}

/*
    对于每个线程，将一个训练周期的相应的lr，hr帧传入训练数据集中
    建立相应的data_loader后即可开始训练
*/
inline void OnlineTraining:: training_threads( int my_rank, 
        std::shared_ptr<MatFrameNumberQueue>& lr_mat_queue, std::shared_ptr<MatFrameNumberQueue>& hr_mat_queue ) {
    
    auto& dataset = this->dataset_list[my_rank];
    for ( int i = 0; i < this->options.training_epoch_num; ) {

        while( !lr_mat_queue->empty() && !hr_mat_queue->empty()
                    && dataset->img_queue.size() < this->options.sampling_cycle / this->options.sample_interval ) {
            
            std::pair< cv::Mat, int > hr_pr;
            std::pair< cv::Mat, int > lr_pr;

            hr_mat_queue->pop( hr_pr );
            lr_mat_queue->pop( lr_pr );

            if ( lr_pr.second != hr_pr.second ) {
                std::cout << "rank:"  << my_rank << "Error: OnlineTraining time slot is missing!" << std::endl;
            }
            cv::Mat lr = lr_pr.first;
            cv::Mat hr = hr_pr.first;
            dataset->push( lr, hr );
        }

        if ( dataset->img_queue.size() <= 0 ) {
            continue;
        }
        this->barrier_dataset.Wait();
        auto train_dataset = dataset->map( torch::data::transforms::Stack<>() );
        auto data_sampler = torch::data::samplers::DistributedRandomSampler(
                                train_dataset.size().value(), this->gpu_count, my_rank, false
                            );
        int total_batch_size = this->options.batch_size;
        int batch_size_per_proc = total_batch_size / this->gpu_count;

        auto train_loader = torch::data::make_data_loader(
                                std::move( train_dataset), data_sampler, batch_size_per_proc
                            );

        auto time_start = Utils::clock();
        OnlineTraining::train_one_epoch( my_rank, *train_loader );
        auto time_end = Utils::clock();

        if( my_rank == 0 ) {
            std::cout <<"OnlineTraining one epoch: " << Utils::get_duration(time_start, time_end) << "ms" << std::endl;
        }
        i++;
    }

    dataset->clear();

}

/*
    在线训练的入口:
        根据给定的可用gpu数目，创建相应的线程，每个线程在一张gpu上训练模型
*/
inline bool OnlineTraining:: training( std::vector<std::shared_ptr<MatFrameNumberQueue>> &lr_mat_queue_list, 
                                        std::vector<std::shared_ptr<MatFrameNumberQueue>> &hr_mat_queue_list) {

    std::vector<std::thread> workers( this->gpu_count );

    auto time1 = Utils::clock();

    for ( int gpu_i = 0; gpu_i < this->gpu_count; gpu_i++ ) {
        workers[gpu_i] = std::thread( &OnlineTraining::training_threads, this, gpu_i, 
            std::ref(lr_mat_queue_list[gpu_i]), std::ref(hr_mat_queue_list[gpu_i]) );
    }
    for (auto& th: workers ) th.join();

    auto time2 = Utils::clock();

    std::cout << "Training once: " << Utils::get_duration(time1, time2) << "ms" << std::endl;
    return true;
}

// 返回模型用于传递给播放端
inline std::string OnlineTraining::  get_model_data() {
    
    std::ostringstream oss;
    this->models[0].save( oss );
    return oss.str();

} 

// 返回训练用的相应的cuda编号
inline std::string OnlineTraining:: get_cuda_num( int idx ) {
    return  "cuda:" + std::to_string( this->options.cuda_num[idx] ) ;
}

#endif // __ONLINETRAINING__