/*
the same with DNS convolution code
*/

#ifndef CAFFE_CCONV_LAYER_HPP_
#define CAFFE_CCONV_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/base_conv_layer.hpp"

namespace caffe {

/**
 * @brief Convolves the input with a bank of pruned filters, 
 *  and (optionally) adds biases.
 */

template <typename Dtype>
class CConvolutionLayer : public BaseConvolutionLayer<Dtype> {
 public:

  explicit CConvolutionLayer(const LayerParameter& param)
      : BaseConvolutionLayer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual inline const char* type() const { return "CConvolution"; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual inline bool reverse_dimensions() { return false; }
  virtual void compute_output_shape();
  
 private:
  Blob<Dtype> weight_tmp_;
  Blob<Dtype> bias_tmp_;   
  Dtype mu_;
  Dtype std_;
  Dtype gamma_;
  Dtype power_; 
  Dtype c_rate_;  
  Dtype alpha_low_;
  Dtype alpha_high_;
  int iter_stop_;
};

}   // namespace caffe

#endif // CAFFE_CCONV_LAYER_HPP_
 