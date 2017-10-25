#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/inq_inner_product_layer.hpp"
#include <cmath>

namespace caffe {

template <typename Dtype>
__global__ void TPCalc(const int n, Dtype *param, Dtype *mask,
                       const Dtype threshold_, const int max_quantum_exp_,
                       const int min_quantum_exp_) {
  CUDA_KERNEL_LOOP(i, n) {
    if (mask[i] == 1) {
      if (param[i] >= threshold_) {
        // exp_ won't be larger than max_quantum_exp_, already checked in the
        int exp_ = floor(log2(4.0 * param[i] / 3.0));
        // CHECK_LE(exp_, max_quantum_exp_) ;
        if (exp_ >= min_quantum_exp_) {
          param[i] = exp2(Dtype(exp_));
          // param[i] = pow(2.0, exp_);
        } else {
          param[i] = 0;
        }
        mask[i] = 0;
      } else if (param[i] <= -threshold_) {
        int exp_ = floor(log2(4.0 * (-param[i]) / 3.0));
        if (exp_ >= min_quantum_exp_) {
          param[i] = -exp2(Dtype(exp_));
        } else {
          param[i] = 0;
        }
        mask[i] = 0;
      }
    }
  }
}

template <typename Dtype>
__global__ void CCMaskApply(const int n, const Dtype *wb, const Dtype *mask,
                            Dtype *wb_t) {
  CUDA_KERNEL_LOOP(index, n) { wb_t[index] = wb[index] * mask[index]; }
}

template <typename Dtype>
void INQInnerProductLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype> *> &bottom, const vector<Blob<Dtype> *> &top) {
  /* for two-power network */
  if (this->phase_ == TRAIN) {
    if (this->iter_ == 0) {
      // Make the corresponding weights & bias into two power form.
      if (this->blobs_.size() == 4 && (this->bias_term_)) {
        LOG(INFO) << "Shaping the weights...";
        ComputeQuantumRange(this->blobs_[0].get(), this->blobs_[2].get(),
                            this->portions_, weight_quantum_values_,
                            num_weight_quantum_values_, max_weight_quantum_exp_,
                            min_weight_quantum_exp_);
        ShapeIntoTwoPower(this->blobs_[0].get(), this->blobs_[2].get(),
                          this->portions_, max_weight_quantum_exp_,
                          min_weight_quantum_exp_);
        LOG(INFO) << "Shaping the bias...";
        ComputeQuantumRange(this->blobs_[1].get(), this->blobs_[3].get(),
                            this->portions_, bias_quantum_values_,
                            num_bias_quantum_values_, max_bias_quantum_exp_,
                            min_bias_quantum_exp_);
        ShapeIntoTwoPower(this->blobs_[1].get(), this->blobs_[3].get(),
                          this->portions_, max_bias_quantum_exp_,
                          min_bias_quantum_exp_);
        LOG(INFO) << "Shaping done in INQ inner_product_layer";
      } else if (this->blobs_.size() == 2 && (!this->bias_term_)) {
        LOG(INFO) << "Warning: No bias terms found... but continue...";
        LOG(INFO) << "Shaping ONLY the weights...";
        ComputeQuantumRange(this->blobs_[0].get(), this->blobs_[1].get(),
                            this->portions_, weight_quantum_values_,
                            num_weight_quantum_values_, max_weight_quantum_exp_,
                            min_weight_quantum_exp_);
        ShapeIntoTwoPower(this->blobs_[0].get(), this->blobs_[1].get(),
                          this->portions_, max_weight_quantum_exp_,
                          min_weight_quantum_exp_);
      }
    }
  }

  const Dtype *weight = this->blobs_[0]->mutable_gpu_data();
  const Dtype *bottom_data = bottom[0]->gpu_data();
  Dtype *top_data = top[0]->mutable_gpu_data();

  if (M_ == 1) {
    caffe_gpu_gemv<Dtype>(CblasNoTrans, N_, K_, (Dtype)1., weight, bottom_data,
                          (Dtype)0., top_data);
    if (bias_term_)
      caffe_gpu_axpy<Dtype>(N_, bias_multiplier_.gpu_data()[0],
                            this->blobs_[1]->gpu_data(), top_data);
  } else {
    caffe_gpu_gemm<Dtype>(CblasNoTrans, transpose_ ? CblasNoTrans : CblasTrans,
                          M_, N_, K_, (Dtype)1., bottom_data, weight, (Dtype)0.,
                          top_data);
    if (bias_term_)
      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, 1, (Dtype)1.,
                            bias_multiplier_.gpu_data(),
                            this->blobs_[1]->gpu_data(), (Dtype)1., top_data);
  }
}

template <typename Dtype>
void INQInnerProductLayer<Dtype>::Backward_gpu(
    const vector<Blob<Dtype> *> &top, const vector<bool> &propagate_down,
    const vector<Blob<Dtype> *> &bottom) {
  // Use the masked weight to propagate back
  const Dtype *top_diff = top[0]->gpu_diff();
  if (this->param_propagate_down_[0]) {
    const Dtype *weightMask = this->blobs_[2]->gpu_data();
    Dtype *weight_diff = this->blobs_[0]->mutable_gpu_diff();
    const Dtype *bottom_data = bottom[0]->gpu_data();
    // Gradient with respect to weight

    CCMaskApply<Dtype><<<CAFFE_GET_BLOCKS(this->blobs_[2]->count()),
                         CAFFE_CUDA_NUM_THREADS>>>(
        this->blobs_[2]->count(), weight_diff, weightMask, weight_diff);
    CUDA_POST_KERNEL_CHECK;

    if (transpose_) {
      caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, K_, N_, M_, (Dtype)1.,
                            bottom_data, top_diff, (Dtype)1.,
                            this->blobs_[0]->mutable_gpu_diff());
    } else {
      caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, N_, K_, M_, (Dtype)1.,
                            top_diff, bottom_data, (Dtype)1.,
                            this->blobs_[0]->mutable_gpu_diff());
    }
  }
  if (bias_term_ && this->param_propagate_down_[1]) {
    const Dtype *biasMask = this->blobs_[3]->gpu_data();
    Dtype *bias_diff = this->blobs_[1]->mutable_gpu_diff();
    // Gradient with respect to bias

    CCMaskApply<Dtype><<<CAFFE_GET_BLOCKS(this->blobs_[3]->count()),
                         CAFFE_CUDA_NUM_THREADS>>>(
        this->blobs_[3]->count(), bias_diff, biasMask, bias_diff);
    CUDA_POST_KERNEL_CHECK;
    caffe_gpu_gemv<Dtype>(CblasTrans, M_, N_, (Dtype)1., top_diff,
                          bias_multiplier_.gpu_data(), (Dtype)1., bias_diff);
  }
  if (propagate_down[0]) {
    // Gradient with respect to bottom data
    if (transpose_) {
      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, K_, N_, (Dtype)1.,
                            top_diff, this->blobs_[0]->gpu_data(), (Dtype)0.,
                            bottom[0]->mutable_gpu_diff());
    } else {
      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, K_, N_, (Dtype)1.,
                            top_diff, this->blobs_[0]->gpu_data(), (Dtype)0.,
                            bottom[0]->mutable_gpu_diff());
    }
  }

  /*
    if (propagate_down[0]) {
      // const	Dtype *weightTmp = this->weight_tmp_.cpu_data();
      // Gradient with respect to bottom data
      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, K_, N_, (Dtype)1.,
                            top_diff, this->blobs_[0]->gpu_data(), (Dtype)0.,
                            bottom[0]->mutable_gpu_diff());
    }
  */
}

template <typename Dtype>
void INQInnerProductLayer<Dtype>::ComputeQuantumRange(
    const Blob<Dtype> *blob, const Blob<Dtype> *blob_mask,
    const vector<float> portions, vector<Dtype> &quantum_values,
    const int &num_quantum_values, int &max_quantum_exp_,
    int &min_quantum_exp_) {

  quantum_values.resize(2 * num_quantum_values + 1);
  const Dtype *values = blob->cpu_data();
  const Dtype *mask = blob_mask->cpu_data();
  Dtype max_value_tobe_quantized = INT_MIN;
  Dtype max_value_quantized = INT_MIN;
  int updated = 0;

  for (unsigned int k = 0; k < blob->count(); ++k) {
    if (mask[k] == 1) {
      if (fabs(values[k]) > max_value_tobe_quantized) {
        max_value_tobe_quantized = fabs(values[k]);
      }
    } else if (mask[k] == 0) {
      if (fabs(values[k]) > max_value_quantized) {
        max_value_quantized = fabs(values[k]);
      }
      ++updated;
    } else {
      LOG(ERROR) << "Mask value is not 0, nor 1, in tp_inner_product_layer";
    }
  }

  if (max_value_quantized != INT_MIN) {
    // normal situation
    CHECK_GT(updated, 0) << "max_value_quantized is not 0.0, but updated is "
                            "0!";
    max_quantum_exp_ = round(log(max_value_quantized) / log(2.0));
    int max_tobe_quantized_exp_ =
        floor(log(4.0 * max_value_tobe_quantized / 3.0) / log(2.0));
    CHECK_GE(max_quantum_exp_, max_tobe_quantized_exp_);
  } else {
    if (updated == 0) {
      // normal situation (nothing quantized yet)
      LOG_IF(INFO, portions_[0] != 0) << "Warning: nothing quantized yet, "
                                         "portion should probably start with "
                                         "0%%!";
      max_quantum_exp_ =
          floor(log(4.0 * max_value_tobe_quantized / 3.0) / log(2.0));
    } else { // DNS model (max_value_quantized ==0 && update != 0)
      max_quantum_exp_ =
          floor(log(4.0 * max_value_tobe_quantized / 3.0) / log(2.0));
    }
  }

  /*
    if (portions[0] == 0) {
      CHECK_EQ(updated, 0) << updated
                           << " updated values while there should be none!";
      max_quantum_exp_ =
          floor(log(4.0 * max_value_tobe_quantized / 3.0) / log(2.0));
    }
    else {
      max_quantum_exp_ = round(log(max_value_quantized) / log(2.0));
      int max_tobe_quantized_exp_ =
          floor(log(4.0 * max_value_tobe_quantized / 3.0) / log(2.0));
      CHECK_LE(max_tobe_quantized_exp_, max_quantum_exp_)
          << "New quantum exp is greater than the one already got!";
    }
  */
  min_quantum_exp_ = max_quantum_exp_ - num_quantum_values + 1;
  std::cout << "Max_power = " << max_quantum_exp_ << std::endl;
  std::cout << "Min_power = " << min_quantum_exp_ << std::endl;
  for (unsigned int k = 0; k < num_quantum_values; ++k) {
    quantum_values[k] = pow(2.0, max_quantum_exp_ - k);
    quantum_values[2 * num_quantum_values - k] = -quantum_values[k];
  }
  quantum_values[num_quantum_values] = 0;
}

template <typename Dtype>
void INQInnerProductLayer<Dtype>::ShapeIntoTwoPower(
    Blob<Dtype> *input_blob, Blob<Dtype> *mask_blob,
    const vector<float> &portions, const int &max_quantum_exp_,
    const int &min_quantum_exp_) {
  const float previous_portion = portions[0];
  const float current_portion = portions[1];
  Dtype *param = input_blob->mutable_gpu_data();
  Dtype *mask = mask_blob->mutable_gpu_data();

  int count = input_blob->count();

  int num_not_yet_quantized = 0;
  vector<Dtype> sorted_param;
  for (int i = 0; i < count; ++i) {
    if (mask[i] == 1) {
      ++num_not_yet_quantized;
      sorted_param.push_back(fabs(param[i]));
    }
  }
  // just an estimation
  int num_init_not_quantized =
      round(Dtype(num_not_yet_quantized) / (1.0 - previous_portion));
  int num_not_tobe_quantized = num_init_not_quantized * (1.0 - current_portion);
  int num_tobe_update = num_not_yet_quantized - num_not_tobe_quantized;

  if (num_tobe_update > 0) {
    sort(sorted_param.begin(), sorted_param.end());
    Dtype threshold_ = sorted_param[num_not_tobe_quantized];
    TPCalc<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, param, mask, threshold_, max_quantum_exp_, min_quantum_exp_);
    CUDA_POST_KERNEL_CHECK;

    LOG(INFO) << "Shaping finished in INQ_conv... [gpu]";
  }

  /*
    for (int i = 0; i < count; ++i) {
      if (mask[i] == 0) {
        updated++;
      }
    }

    int left = count - updated;
    int update = floor(count * current_portion) - updated;

    vector<Dtype> sort_param(left);

    int k = 0;
    if (update > 0) {
      for (int i = 0; i < count; ++i) {
        if (mask[i] == 1) {
          sort_param[k++] = fabs(param[i]);
        }
      }
      CHECK_EQ(k, left) << "Num of weights/bias that are not in 2 power form "
                           "does NOT match the portion!";
      sort(sort_param.begin(), sort_param.end());
      Dtype threshold = sort_param[left - update];

      TPCalc<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
          count, param, mask, threshold);
      CUDA_POST_KERNEL_CHECK;

      LOG(INFO) << "Shaping finished in tp_inner... [gpu]";
    }
  */
}

INSTANTIATE_LAYER_GPU_FUNCS(INQInnerProductLayer);

} // namespace caffe
