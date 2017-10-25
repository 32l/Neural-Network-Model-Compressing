#ifndef INQ_INNER_PRODUCT_LAYER_HPP_
#define INQ_INNER_PRODUCT_LAYER_HPP_

#include <vector>
#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

template <typename Dtype> 
class INQInnerProductLayer : public Layer<Dtype> {
public:
  explicit INQInnerProductLayer(const LayerParameter &param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype> *> &bottom,
                          const vector<Blob<Dtype> *> &top);
  virtual void Reshape(const vector<Blob<Dtype> *> &bottom,
                       const vector<Blob<Dtype> *> &top);

  virtual inline const char *type() const { return "INQInnerProduct"; }
  virtual inline int ExactNumBottomBlobs() const { return 1; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

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
  int M_;
  int K_; // count or size of each output (channel * height * weight )
  int N_; // number of output
  bool bias_term_;
  Blob<Dtype> bias_multiplier_;
  bool transpose_;  ///< if true, assume transposed weights

  // INQ functions & variables
  virtual void ComputeQuantumRange(const Blob<Dtype> *blob,
                                   const Blob<Dtype> *mask,
                                   const vector<float> portions,
                                   vector<Dtype> &quantum_values,
                                   const int &num_quantum_values,
                                   int &max_quantum_exp_,
                                   int &min_quantum_exp_);
  virtual void ShapeIntoTwoPower_cpu(Blob<Dtype> *input_blob,
                                 Blob<Dtype> *mask_blob,
                                 const vector<float> &portions,
                                 const int &max_quantum_exp_,
                                 const int &min_quantum_exp_);
  virtual void ShapeIntoTwoPower_gpu(Blob<Dtype> *input_blob,
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

#endif // INQ_INNER_PRODUCT_LAYER_HPP_
