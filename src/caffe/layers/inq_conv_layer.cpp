#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/inq_conv_layer.hpp"
#include <cmath>
#include <float.h>

namespace caffe {
template <typename Dtype>
void INQConvolutionLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype> *> &bottom,
                                            const vector<Blob<Dtype> *> &top) {
  BaseConvolutionLayer<Dtype>::LayerSetUp(bottom, top);
  /********** for neural network model compression **********/
  INQConvolutionParameter inq_conv_param =
      this->layer_param_.inq_convolution_param();
  // expand blobs_ size
  if (this->blobs_.size() == 2 && (this->bias_term_)) {
    this->blobs_.resize(4);
    // Intialize and fill the weight mask & bias mask
    this->blobs_[2].reset(new Blob<Dtype>(this->blobs_[0]->shape()));
    shared_ptr<Filler<Dtype> > weight_mask_filler(
        GetFiller<Dtype>(inq_conv_param.weight_mask_filler()));
    weight_mask_filler->Fill(this->blobs_[2].get());
    this->blobs_[3].reset(new Blob<Dtype>(this->blobs_[1]->shape()));
    shared_ptr<Filler<Dtype> > bias_mask_filler(
        GetFiller<Dtype>(inq_conv_param.bias_mask_filler()));
    bias_mask_filler->Fill(this->blobs_[3].get());
  } else if (this->blobs_.size() == 1 && (!this->bias_term_)) {
    this->blobs_.resize(2);
    // Intialize and fill the weight mask
    this->blobs_[1].reset(new Blob<Dtype>(this->blobs_[0]->shape()));
    shared_ptr<Filler<Dtype> > bias_mask_filler(
        GetFiller<Dtype>(inq_conv_param.bias_mask_filler()));
    bias_mask_filler->Fill(this->blobs_[1].get());
  }
  // Initialize the portion array
  this->num_portions_ = inq_conv_param.portion().size();
  CHECK_GT(this->num_portions_, 0)
      << "Number of portions must be greater than 0";
  this->portions_.resize(this->num_portions_);
  for (int i = 0; i < this->num_portions_; ++i) {
    portions_[i] = inq_conv_param.portion(i);
  }
  CHECK_LE(portions_[0], portions_[1]);
  // Get the number of quantum values
  this->num_weight_quantum_values_ = inq_conv_param.num_quantum_values();
  this->num_bias_quantum_values_ = inq_conv_param.num_quantum_values();
  /**********************************************************/
}

template <typename Dtype>
void INQConvolutionLayer<Dtype>::compute_output_shape() {
  const int *kernel_shape_data = this->kernel_shape_.cpu_data();
  const int *stride_data = this->stride_.cpu_data();
  const int *pad_data = this->pad_.cpu_data();
  const int *dilation_data = this->dilation_.cpu_data();
  this->output_shape_.clear();
  for (int i = 0; i < this->num_spatial_axes_; ++i) {
    // i + 1 to skip channel axis
    const int input_dim = this->input_shape(i + 1);
    const int kernel_extent = dilation_data[i] * (kernel_shape_data[i] - 1) + 1;
    const int output_dim =
        (input_dim + 2 * pad_data[i] - kernel_extent) / stride_data[i] + 1;
    this->output_shape_.push_back(output_dim);
  }
}

template <typename Dtype>
void INQConvolutionLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype> *> &bottom, const vector<Blob<Dtype> *> &top) {
  /********** for neural network model compression **********/
  if (this->phase_ == TRAIN) {
    if (this->iter_ == 0) {
      // Make the corresponding weights & bias into two power form.
      if (this->blobs_.size() == 4 && (this->bias_term_)) {
        std::cout << "Shaping the weights..." << std::endl;
        ComputeQuantumRange(this->blobs_[0].get(), this->blobs_[2].get(),
                            this->portions_, weight_quantum_values_,
                            num_weight_quantum_values_, max_weight_quantum_exp_,
                            min_weight_quantum_exp_);
        ShapeIntoTwoPower(this->blobs_[0].get(), this->blobs_[2].get(),
                          this->portions_, max_weight_quantum_exp_,
                          min_weight_quantum_exp_);
        std::cout << "Shaping the bias..." << std::endl;
        ComputeQuantumRange(this->blobs_[1].get(), this->blobs_[3].get(),
                            this->portions_, bias_quantum_values_,
                            num_bias_quantum_values_, max_bias_quantum_exp_,
                            min_bias_quantum_exp_);
        ShapeIntoTwoPower(this->blobs_[1].get(), this->blobs_[3].get(),
                          this->portions_, max_bias_quantum_exp_,
                          min_bias_quantum_exp_);
      } else if (this->blobs_.size() == 2 && (!this->bias_term_)) {
        LOG(INFO) << "ERROR: No bias terms found... but continue...";
        ComputeQuantumRange(this->blobs_[0].get(), this->blobs_[1].get(),
                            this->portions_, weight_quantum_values_,
                            num_weight_quantum_values_, max_weight_quantum_exp_,
                            min_weight_quantum_exp_);
        std::cout << "Shaping ONLY the weights..." << std::endl;
        ShapeIntoTwoPower(this->blobs_[0].get(), this->blobs_[1].get(),
                          this->portions_, max_weight_quantum_exp_,
                          min_weight_quantum_exp_);
      }
    }
  }
  /**********************************************************/

  const Dtype *weight = this->blobs_[0]->cpu_data();
  const Dtype *bias = NULL;
  if (this->bias_term_) {
    bias = this->blobs_[1]->cpu_data();
  }
  // Forward calculation with quantized weight and bias
  for (int i = 0; i < bottom.size(); ++i) {
    const Dtype *bottom_data = bottom[i]->cpu_data();
    Dtype *top_data = top[i]->mutable_cpu_data();
    for (int n = 0; n < this->num_; ++n) {
      this->forward_cpu_gemm(bottom_data + bottom[i]->offset(n), weight,
                             top_data + top[i]->offset(n));
      if (this->bias_term_) {
        this->forward_cpu_bias(top_data + top[i]->offset(n), bias);
      }
    }
  }
}

template <typename Dtype>
void INQConvolutionLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype> *> &top, const vector<bool> &propagate_down,
    const vector<Blob<Dtype> *> &bottom) {
  // const Dtype *weightTmp = this->weight_tmp_.cpu_data();
  const Dtype *weight = this->blobs_[0]->cpu_data();
  const Dtype *weightMask = this->blobs_[2]->cpu_data();
  Dtype *weight_diff = this->blobs_[0]->mutable_cpu_diff();
  for (int i = 0; i < top.size(); ++i) {
    const Dtype *top_diff = top[i]->cpu_diff();
    // Bias gradient, if necessary.
    if (this->bias_term_ && this->param_propagate_down_[1]) {
      const Dtype *biasMask = this->blobs_[3]->cpu_data();
      Dtype *bias_diff = this->blobs_[1]->mutable_cpu_diff();
      for (unsigned int k = 0; k < this->blobs_[1]->count(); ++k) {
        bias_diff[k] = bias_diff[k] * biasMask[k];
      }
      for (int n = 0; n < this->num_; ++n) {
        this->backward_cpu_bias(bias_diff, top_diff + top[i]->offset(n));
      }
    }
    if (this->param_propagate_down_[0] || propagate_down[i]) {
      const Dtype *bottom_data = bottom[i]->cpu_data();
      Dtype *bottom_diff = bottom[i]->mutable_cpu_diff();
      for (unsigned int k = 0; k < this->blobs_[0]->count(); ++k) {
        weight_diff[k] = weight_diff[k] * weightMask[k];
      }
      for (int n = 0; n < this->num_; ++n) {
        // gradient w.r.t. weight. Note that we will accumulate diffs.
        if (this->param_propagate_down_[0]) {
          this->weight_cpu_gemm(bottom_data + bottom[i]->offset(n),
                                top_diff + top[i]->offset(n), weight_diff);
        }
        // gradient w.r.t. bottom data, if necessary.
        if (propagate_down[i]) {
          this->backward_cpu_gemm(top_diff + top[i]->offset(n), weight,
                                  bottom_diff + bottom[i]->offset(n));
        }
      }
    }
  }
}

template <typename Dtype>
void INQConvolutionLayer<Dtype>::ComputeQuantumRange(
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
      LOG(ERROR) << "Mask value must be either 0 or 1 !";
    }
  }
  

  if (fabs(max_value_quantized) <= FLT_EPSILON) { // DNS init model
    CHECK_GT(updated, 0) << "max_value_quantized(" << max_value_quantized 
                         << ") is not 0.0, but updated is 0!";
    CHECK_GT(max_value_tobe_quantized, FLT_EPSILON) << "error wiht DNS raw model!";
    max_quantum_exp_ = floor(log(4.0 * max_value_tobe_quantized / 3.0) / log(2.0));
  } else if (max_value_quantized == INT_MIN) { // normal init model
    CHECK_EQ(updated, 0) << "Normal init model, updated should be 0!";
    CHECK_GT(max_value_tobe_quantized, FLT_EPSILON) << "error wiht normal init model!";
    max_quantum_exp_ = floor(log(4.0 * max_value_tobe_quantized / 3.0) / log(2.0));
  } else { // normal situation, both quantized and not quantized exist
    CHECK_GT(max_value_tobe_quantized, FLT_EPSILON);
    CHECK_GT(max_value_quantized, max_value_tobe_quantized);
    max_quantum_exp_ = round(log(max_value_quantized) / log(2.0));
  }
    

/*
  if (max_value_quantized != INT_MIN) {
    // normal situation
    CHECK_GT(updated, 0) << "max_value_quantized(" << max_value_quantized <<
                            ") is not 0.0, but updated is "
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
*/
  /*
    if (portions[0] == 0) {
      CHECK_EQ(updated, 0) << updated
                           << " updated values while there should be none!";
      max_quantum_exp_ =
          floor(log(4.0 * max_value_tobe_quantized / 3.0) / log(2.0));
    } else {
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
void INQConvolutionLayer<Dtype>::ShapeIntoTwoPower(
    Blob<Dtype> *input_blob, Blob<Dtype> *mask_blob,
    const vector<float> &portions, const int &max_quantum_exp_,
    const int &min_quantum_exp_) {
  const float previous_portion = portions[0];
  const float current_portion = portions[1];
  if (current_portion == 0) {
    return;
  }
  Dtype *param = input_blob->mutable_cpu_data();
  Dtype *mask = mask_blob->mutable_cpu_data();
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
  int num_not_tobe_quantized =
      round(num_init_not_quantized * (1.0 - current_portion));
  int num_tobe_update = num_not_yet_quantized - num_not_tobe_quantized;

  LOG(INFO) << "portions: " << previous_portion * 100 <<"% -> "
            << current_portion * 100 << "% ("
            << "total: " 
            << Dtype(count-num_not_yet_quantized)/count*100 << "% -> "
            << Dtype(count-num_not_tobe_quantized)/count*100<< "%"
            << ")";
  LOG(INFO) << "init_not_quantized/total: "
            << num_init_not_quantized << "/" 
            << count;            
  LOG(INFO) << "to_update/not_tobe_quantized/not_yet_quantized: " 
            << num_tobe_update << "/"
            << num_not_tobe_quantized << "/"
            << num_not_yet_quantized ;
            
  if (num_tobe_update > 0) {
    sort(sorted_param.begin(), sorted_param.end());
    Dtype threshold_ = sorted_param[num_not_tobe_quantized];
    for (int i = 0; i < count; ++i) {
      if (mask[i] == 1) {
        if (param[i] >= threshold_) {
          // exp_ won't be larger than max_quantum_exp_, already checked in the
          // ComputeQuantumRange()
          int exp_ = floor(log(4.0 * param[i] / 3.0) / log(2.0));
          // CHECK_LE(exp_, max_quantum_exp_) ;
          if (exp_ >= min_quantum_exp_) {
            param[i] = pow(2.0, exp_);
          } else {
            param[i] = 0;
          }
          mask[i] = 0;
        } else if (param[i] <= -threshold_) {
          int exp_ = floor(log(4.0 * (-param[i]) / 3.0) / log(2.0));
          if (exp_ >= min_quantum_exp_) {
            param[i] = -pow(2.0, exp_);
          } else {
            param[i] = 0;
          }
          mask[i] = 0;
        }
      }
    }
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
      for (int i = 0; i < count; ++i) {
        if (mask[i] == 1) {
          if (param[i] >= threshold) {
            // exp_ won't be larger than max_quantum_exp_, already checked in
  the
            // ComputeQuantumRange()
            int exp_ = floor(log(4.0 * param[i] / 3.0) / log(2.0));
            // CHECK_LE(exp_, max_quantum_exp_) ;
            if (exp_ >= min_quantum_exp_) {
              param[i] = pow(2.0, exp_);
            }
            else {
              param[i] = 0;
            }
            mask[i] = 0;
          }
          else if (param[i] <= -threshold) {
            int exp_ = floor(log(4.0 * (-param[i]) / 3.0) / log(2.0));
            if (exp_ >= min_quantum_exp_) {
              param[i] = -pow(2.0, exp_);
            }
            else {
              param[i] = 0;
            }
            mask[i] = 0;
          }
        }
      }
    }
  }
  */
}

#ifdef CPU_ONLY
STUB_GPU(INQConvolutionLayer);
#endif

INSTANTIATE_CLASS(INQConvolutionLayer);
REGISTER_LAYER_CLASS(INQConvolution);

} // namespace caffe
