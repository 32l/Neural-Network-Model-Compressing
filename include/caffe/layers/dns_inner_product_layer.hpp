#ifndef CAFFE_DNS_INNER_PRODUCT_LAYER_HPP_
#define CAFFE_DNS_INNER_PRODUCT_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe{
/**
 * @brief The DNS InnerProduct layer, also known as a DNS 
 *  "fully-connected" layer
 *
 * TODO(dox): thorough documentation for Forward, Backward, and proto params.
 */
template <typename Dtype>
class DNSInnerProductLayer : public Layer<Dtype> {
 public:
  explicit DNSInnerProductLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "DNSInnerProduct"; }
  virtual inline int ExactNumBottomBlobs() const { return 1; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  int M_;
  int K_;
  int N_;
  bool bias_term_;
  Blob<Dtype> bias_multiplier_;
  bool transpose_;  ///< if true, assume transposed weights

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

#endif  // CAFFE_DNSINNER_PRODUCT_LAYER_HPP_
