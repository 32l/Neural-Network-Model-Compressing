#include <string>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/sgd_solvers.hpp"

namespace caffe {

#ifndef CPU_ONLY
template <typename Dtype>
void sgd_update_gpu(int N, Dtype *g, Dtype *h, Dtype momentum,
                    Dtype local_rate);
template <typename Dtype>
void inq_lars_sgd_update_gpu(int N, const Dtype *mask, Dtype *g, Dtype *h, 
                    Dtype momentum, Dtype local_rate);
#endif

template<typename Dtype>
Dtype LarsSGDSolver<Dtype>::GetLocalRate(int param_id) const {
  Dtype local_lr = 1.0;
  // if (this->param_.local_lr_auto()) {
  Blob<Dtype>* param = this->net_->learnable_params()[param_id];
  const Dtype w_norm_ = std::sqrt(param->sumsq_data());
  const Dtype wgrad_norm_ = std::sqrt(param->sumsq_diff());
  const Dtype gw_ratio = this->param_.local_gw_ratio();
  Dtype weight_decay = this->param_.weight_decay();
  if (w_norm_ > 0.F && wgrad_norm_ >  0.F) {
    local_lr = gw_ratio * w_norm_ / (wgrad_norm_ + weight_decay * w_norm_);
  }
/*
#ifdef DEBUG
    if (Caffe::root_solver()
        && this->param_.display()
        && (this->iter_ % this->param_.display() == 0)) {
      const int layer_id = this->net_->param_layer_indices(param_id).first;
      const string& layer_name = this->net_->layer_names()[layer_id];
      const int blob_id  = this->net_->param_layer_indices(param_id).second;
      LOG(INFO) << layer_name <<"."<< blob_id << " lr=" << local_lr
                << " " << "\t  w=" << w_norm << "\t dw=" << wgrad_norm;
    }
#endif
*/
  // }
  return local_lr;
}


template <typename Dtype>
void LarsSGDSolver<Dtype>::ComputeUpdateValue(int param_id, Dtype rate) {
  const vector<Blob<Dtype> *> &net_params = this->net_->learnable_params();
  // const vector<float> &net_params_lr = this->net_->params_lr();
  Dtype momentum = this->param_.momentum();
  // Dtype local_rate = rate * net_params_lr[param_id];
  Dtype local_rate = rate * GetLocalRate(param_id);
  /********** for neural network model compression **********/
  int blobs_to_skip = 2;
  // const vector<int> &mask_param_ids_ = this->net_->mask_param_ids();
  const vector<int> &learnable_param_ids_ = this->net_->learnable_param_ids();
  const vector<int> &inq_param_ids_ = this->net_->inq_param_ids();
  bool is_inq_param_;
  is_inq_param_ =
      std::find(inq_param_ids_.begin(), inq_param_ids_.end(),
                learnable_param_ids_[param_id]) != inq_param_ids_.end();
  /**********************************************************/

  // Compute the update to history, then copy it to the parameter diff.
  switch (Caffe::mode()) {
  case Caffe::CPU: {
    caffe_cpu_axpby(net_params[param_id]->count(), local_rate,
                    net_params[param_id]->cpu_diff(), momentum,
                    this->history_[param_id]->mutable_cpu_data());

    /********** for neural network model compression **********/
    // if(std::find(mask_param_ids_.begin(), mask_param_ids_.end(),
    // learnable_param_ids_[param_id + blobs_to_skip]) != mask_param_ids_.end())
    // {
    if (is_inq_param_) {
      // std::cout << "Using a mask layer ..." << std::endl;
      CHECK_EQ(net_params[param_id]->count(),
               net_params[param_id + blobs_to_skip]->count())
          << "Blobs' count should be the same with its Mask' count !!";
      // history_diff is an alias for history_
      Dtype *history_diff = this->history_[param_id]->mutable_cpu_data();
      const Dtype *mask = net_params[param_id + blobs_to_skip]->cpu_data();
      caffe_copy(net_params[param_id]->count(), this->history_[param_id]->cpu_data(),
                 this->temp_[param_id]->mutable_cpu_data());
      caffe_mul(net_params[param_id]->count(), this->temp_[param_id]->cpu_data(),
                mask, history_diff);
    }
    /**********************************************************/

    caffe_copy(net_params[param_id]->count(), this->history_[param_id]->cpu_data(),
               net_params[param_id]->mutable_cpu_diff());
    break;
  }
  case Caffe::GPU: {
#ifndef CPU_ONLY
    // LOG(INFO) << "is_inq : " << is_inq_param_ ;
    if (!is_inq_param_) {
      // normal SGD update
      sgd_update_gpu(net_params[param_id]->count(),
                     net_params[param_id]->mutable_gpu_diff(),
                     this->history_[param_id]->mutable_gpu_data(), momentum,
                     local_rate);
    } 
    /********** for neural network model compression **********/
    else {
      //LOG(INFO) <<"right before inq_sgd_update_gpu";
      inq_lars_sgd_update_gpu(net_params[param_id]->count(),
                         // inq mask blob
                         net_params[param_id + blobs_to_skip]->gpu_data(),
                         net_params[param_id]->mutable_gpu_diff(),
                         this->history_[param_id]->mutable_gpu_data(), momentum,
                         local_rate);
      // LOG(INFO) <<"right after inq_sgd_update_gpu";
    }
    /**********************************************************/
#else
    NO_GPU;
#endif
    break;
  }
  default:
    LOG(FATAL) << "Unknown caffe mode: " << Caffe::mode();
  }
}

INSTANTIATE_CLASS(LarsSGDSolver);
REGISTER_SOLVER_CLASS(LarsSGD);

} // namespace caffe
