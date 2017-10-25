#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/inq_inner_product_layer.hpp"
#include <cmath>

/*
#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/math_functions.hpp"
#include <cmath>

*/

namespace caffe {
template <typename Dtype>
void INQInnerProductLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype> *> &bottom, const vector<Blob<Dtype> *> &top) {
  const int num_output = this->layer_param_.inner_product_param().num_output();
  bias_term_ = this->layer_param_.inner_product_param().bias_term();
  N_ = num_output;
  const int axis = bottom[0]->CanonicalAxisIndex(
      this->layer_param_.inner_product_param().axis());
  K_ = bottom[0]->count(axis);
  // Check if we need to set up the weights
  if (this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    if (this->bias_term_) {
      this->blobs_.resize(2);
    } else {
      this->blobs_.resize(1);
    }
    // Intialize the weight
    vector<int> weight_shape(2);
    weight_shape[0] = N_;
    weight_shape[1] = K_;
    this->blobs_[0].reset(new Blob<Dtype>(weight_shape));
    // fill the weights
    shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
        this->layer_param_.inner_product_param().weight_filler()));
    weight_filler->Fill(this->blobs_[0].get());
    // If necessary, intiialize and fill the bias term
    if (this->bias_term_) {
      vector<int> bias_shape(1, N_);
      this->blobs_[1].reset(new Blob<Dtype>(bias_shape));
      shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype>(
          this->layer_param_.inner_product_param().bias_filler()));
      bias_filler->Fill(this->blobs_[1].get());
    }
  } // parameter initialization
  this->param_propagate_down_.resize(this->blobs_.size(), true);

  /********** for neural network model compression **********/
  INQInnerProductParameter inq_inner_param =
      this->layer_param_.inq_inner_product_param();
  // Initialize the portion array
  this->num_portions_ = inq_inner_param.portion().size();
  CHECK_GT(this->num_portions_, 0)
      << "Number of portions must be greater than 0.";
  this->portions_.resize(this->num_portions_);
  for (int i = 0; i < this->num_portions_; ++i) {
    portions_[i] = inq_inner_param.portion(i);
  }
  CHECK_LE(portions_[0], portions_[1]);

  // Initialize the mask
  if (this->blobs_.size() == 2 && (this->bias_term_)) {
    this->blobs_.resize(4);
    // Intialize and fill the weight_mask & bias_mask
    this->blobs_[2].reset(new Blob<Dtype>(this->blobs_[0]->shape()));
    shared_ptr<Filler<Dtype> > weight_mask_filler(
        GetFiller<Dtype>(inq_inner_param.weight_mask_filler()));
    weight_mask_filler->Fill(this->blobs_[2].get());
    this->blobs_[3].reset(new Blob<Dtype>(this->blobs_[1]->shape()));
    shared_ptr<Filler<Dtype> > bias_mask_filler(
        GetFiller<Dtype>(inq_inner_param.bias_mask_filler()));
    bias_mask_filler->Fill(this->blobs_[3].get());
  } else if (this->blobs_.size() == 1 && (!this->bias_term_)) {
    this->blobs_.resize(2);
    // Intialize and fill the weight_mask
    this->blobs_[1].reset(new Blob<Dtype>(this->blobs_[0]->shape()));
    shared_ptr<Filler<Dtype> > weight_mask_filler(
        GetFiller<Dtype>(inq_inner_param.weight_mask_filler()));
    weight_mask_filler->Fill(this->blobs_[1].get());
  }
  // Get the max power
  this->num_weight_quantum_values_ = inq_inner_param.num_quantum_values();
  this->num_bias_quantum_values_ = inq_inner_param.num_quantum_values();
  /**********************************************************/
}

template <typename Dtype>
void INQInnerProductLayer<Dtype>::Reshape(const vector<Blob<Dtype> *> &bottom,
                                          const vector<Blob<Dtype> *> &top) {
  // Figure out the dimensions
  const int axis = bottom[0]->CanonicalAxisIndex(
      this->layer_param_.inner_product_param().axis());
  const int new_K = bottom[0]->count(axis);
  CHECK_EQ(K_, new_K)
      << "Input size incompatible with inner product parameters.";
  // The first "axis" dimensions are independent inner products; the total
  // number of these is M_, the product over these dimensions.
  M_ = bottom[0]->count(0, axis);
  // The top shape will be the bottom shape with the flattened axes dropped,
  // and replaced by a single axis with dimension num_output (N_).
  vector<int> top_shape = bottom[0]->shape();
  top_shape.resize(axis + 1);
  top_shape[axis] = N_;
  top[0]->Reshape(top_shape);
  // Set up the bias multiplier
  if (bias_term_) {
    vector<int> bias_shape(1, M_);
    bias_multiplier_.Reshape(bias_shape);
    caffe_set(M_, Dtype(1), bias_multiplier_.mutable_cpu_data());
  }
}

template <typename Dtype>
void INQInnerProductLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype> *> &bottom, const vector<Blob<Dtype> *> &top) {
  /********** for neural network model compression **********/
  if (this->phase_ == TRAIN) {
    if (this->iter_ == 0) {
      // Make the corresponding weights & bias into two power form.
      if (this->blobs_.size() == 4 && (this->bias_term_)) {
        ComputeQuantumRange(this->blobs_[0].get(), this->blobs_[2].get(),
                            this->portions_, weight_quantum_values_,
                            num_weight_quantum_values_, max_weight_quantum_exp_,
                            min_weight_quantum_exp_);
        std::cout << "Shaping the weights..." << std::endl;
        ShapeIntoTwoPower(this->blobs_[0].get(), this->blobs_[2].get(),
                          this->portions_, max_weight_quantum_exp_,
                          min_weight_quantum_exp_);

        ComputeQuantumRange(this->blobs_[1].get(), this->blobs_[3].get(),
                            this->portions_, bias_quantum_values_,
                            num_bias_quantum_values_, max_bias_quantum_exp_,
                            min_bias_quantum_exp_);
        std::cout << "Shaping the bias..." << std::endl;
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
  const Dtype *weight = this->blobs_[0]->mutable_cpu_data();
  const Dtype *bias = NULL;
  if (this->bias_term_) {
    bias = this->blobs_[1]->mutable_cpu_data();
  }
  // Forward calculation
  const Dtype *bottom_data = bottom[0]->cpu_data();
  Dtype *top_data = top[0]->mutable_cpu_data();
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, N_, K_, (Dtype)1.,
                        bottom_data, weight, (Dtype)0., top_data);
  if (bias_term_) {
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, 1, (Dtype)1.,
                          bias_multiplier_.cpu_data(), bias, (Dtype)1.,
                          top_data);
  }
}

template <typename Dtype>
void INQInnerProductLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype> *> &top, const vector<bool> &propagate_down,
    const vector<Blob<Dtype> *> &bottom) {
  // Use the quantized weight to propagate back
  const Dtype *top_diff = top[0]->cpu_diff();
  if (this->param_propagate_down_[0]) {
    const Dtype *weightMask = this->blobs_[2]->cpu_data();
    Dtype *weight_diff = this->blobs_[0]->mutable_cpu_diff();
    const Dtype *bottom_data = bottom[0]->cpu_data();
    // Gradient with respect to weight
    for (unsigned int k = 0; k < this->blobs_[0]->count(); ++k) {
      weight_diff[k] = weight_diff[k] * weightMask[k];
    }
    caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, N_, K_, M_, (Dtype)1.,
                          top_diff, bottom_data, (Dtype)1., weight_diff);
  }
  if (bias_term_ && this->param_propagate_down_[1]) {
    const Dtype *biasMask = this->blobs_[3]->cpu_data();
    Dtype *bias_diff = this->blobs_[1]->mutable_cpu_diff();
    // Gradient with respect to bias
    for (unsigned int k = 0; k < this->blobs_[1]->count(); ++k) {
      bias_diff[k] = bias_diff[k] * biasMask[k];
    }
    caffe_cpu_gemv<Dtype>(CblasTrans, M_, N_, (Dtype)1., top_diff,
                          bias_multiplier_.cpu_data(), (Dtype)1., bias_diff);
  }
  if (propagate_down[0]) {
    // Gradient with respect to bottom data
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, K_, N_, (Dtype)1.,
                          top_diff, this->blobs_[0]->cpu_data(), (Dtype)0.,
                          bottom[0]->mutable_cpu_diff());
  }
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
                                       "portions should probably start with "
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
void INQInnerProductLayer<Dtype>::ShapeIntoTwoPower(
    Blob<Dtype> *input_blob, Blob<Dtype> *mask_blob,
    const vector<float> &portions, const int &max_quantum_exp_,
    const int &min_quantum_exp_) {
  // Get the portions
  const float previous_portion = portions[0];
  const float current_portion = portions[1];
  if (current_portion == 0) {
    return;
  }
  // Get the number of quantized (updated) parameters
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
            << Dtype(count-num_not_tobe_quantized)/count * 100 << "%"
            << ")";
  LOG(INFO) << "init_not_quantized/total: "
            << num_init_not_quantized << "/" 
            << count;            
  LOG(INFO) << "to_update/not_tobe_quantized/not_yet_quantized: " 
            << num_tobe_update << "/"
            << num_not_tobe_quantized << "/"
            << num_not_yet_quantized ;

  // LOG(INFO) <<"to_update/not_yet_quantized/total:
  // "<<num_tobe_update<<num_not_yet_quantized<<count;
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
    // number of parameters that need to be updated
    int update = floor(count * current_portion) - updated;
    vector<Dtype> sort_param(left);
    int k = 0;
    // Start quantization
    if (update > 0) {
      // sort parameters according to absolute value
      for (int i = 0; i < count; ++i) {
        if (mask[i] == 1) {
          sort_param[k++] = fabs(param[i]);
        }
      }
      CHECK_EQ(k, left) << "Num of weights/bias that are not in 2 power form "
                           "does NOT match the portion!";
      sort(sort_param.begin(), sort_param.end());
      // quantization threshold
      Dtype threshold = sort_param[left - update];
      // check and quantize each parameters
      for (int i = 0; i < count; ++i) {
        if (mask[i] == 1) {
          if (param[i] >= threshold) {
            // exp_ won't be larger than max_quantum_exp_, already checked in
    the
            // ComputeQuantumRange()
            int exp_ = floor(log(4.0 * param[i] / 3.0) / log(2.0));
            if (exp_ >= min_quantum_exp_) {
              param[i] = pow(2.0, exp_);
            } else {
              param[i] = 0;
            }
            mask[i] = 0;
          } else if (param[i] <= -threshold) {
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
  */

} // ShapeIntoTwoPower()

#ifdef CPU_ONLY
STUB_GPU(INQInnerProductLayer);
#endif

INSTANTIATE_CLASS(INQInnerProductLayer);
REGISTER_LAYER_CLASS(INQInnerProduct);

} // namespace caffe
