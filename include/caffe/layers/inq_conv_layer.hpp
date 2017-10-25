#ifndef INQ_CONV_LAYER_HPP_
#define INQ_CONV_LAYER_HPP_

#include <vector>
#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/layers/base_conv_layer.hpp"

namespace caffe {
template <typename Dtype>
class INQConvolutionLayer : public BaseConvolutionLayer<Dtype> {
public:
  explicit INQConvolutionLayer(const LayerParameter &param)
      : BaseConvolutionLayer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype> *> &bottom,
                          const vector<Blob<Dtype> *> &top);
  virtual inline const char *type() const { return "INQConvolution"; }

protected:
  virtual void Forward_cpu(const vector<Blob<Dtype> *> &bottom,
                           const vector<Blob<Dtype> *> &top);
  virtual void Forward_gpu(const vector<Blob<Dtype> *> &bottom,
                           const vector<Blob<Dtype> *> &top);
  virtual void Backward_cpu(const vector<Blob<Dtype> *> &top,
                            const vector<bool> &propagate_down,
                            const vector<Blob<Dtype> *> &bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype> *> &top,
                            const vector<bool> &propagate_down,
                            const vector<Blob<Dtype> *> &bottom);
  virtual inline bool reverse_dimensions() { return false; }
  virtual void compute_output_shape();

  // INQ functions & variables
  virtual void ComputeQuantumRange(const Blob<Dtype> *blob,
                                   const Blob<Dtype> *mask,
                                   const vector<float> portions,
                                   vector<Dtype> &quantum_values,
                                   const int &num_quantum_values,
                                   int &max_quantum_exp_,
                                   int &min_quantum_exp_);
  virtual void ShapeIntoTwoPower_cpu( Blob<Dtype> *input_blob,
                                      Blob<Dtype> *mask_blob,
                                      const vector<float> &portions,
                                      const int &max_quantum_exp_,
                                      const int &min_quantum_exp_);
  virtual void ShapeIntoTwoPower_gpu( Blob<Dtype> *input_blob,
                                      Blob<Dtype> *mask_blob,
                                      const vector<float> &portions,
                                      const int &max_quantum_exp_,
                                      const int &min_quantum_exp_);                   
                                 
  vector<float> portions_;
  int num_portions_;
  int num_weight_quantum_values_;
  int num_bias_quantum_values_;
  int max_weight_quantum_exp_;
  int min_weight_quantum_exp_;
  int max_bias_quantum_exp_;
  int min_bias_quantum_exp_;
  vector<Dtype> weight_quantum_values_;
  vector<Dtype> bias_quantum_values_;
};

} // namespace caffe

#endif // INQ_CONV_LAYER_HPP_
