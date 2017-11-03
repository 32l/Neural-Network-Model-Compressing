#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void SGDUpdate(int N, Dtype *g, Dtype *h, Dtype momentum,
                          Dtype local_rate) {
  CUDA_KERNEL_LOOP(i, N) { g[i] = h[i] = momentum * h[i] + local_rate * g[i]; }
}

template <typename Dtype>
__global__ void INQ_SGDUpdate(int N, const Dtype *mask, Dtype *g, Dtype *h,
                              Dtype momentum, Dtype local_rate) {
  CUDA_KERNEL_LOOP(i, N) {
    g[i] = h[i] = mask[i] * (momentum * h[i] + local_rate * g[i]);
  }
}
/*
  sgd_update_gpu(net_params[param_id]->count(),
      net_params[param_id]->mutable_gpu_diff(),
      history_[param_id]->mutable_gpu_data(),
      momentum, local_rate);
*/
template <typename Dtype>
void sgd_update_gpu(int N, Dtype *g, Dtype *h, Dtype momentum,
                    Dtype local_rate) {
  SGDUpdate<Dtype> // NOLINT_NEXT_LINE(whitespace/operators)
      <<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(N, g, h, momentum,
                                                        local_rate);
  CUDA_POST_KERNEL_CHECK;
}

/*
inq_sgd_update_gpu(net_params[param_id]->count(),
                   net_params[param_id + blobs_to_skip]->gpu_data(),
                   net_params[param_id]->mutable_gpu_diff(),
                   history_[param_id]->mutable_gpu_data(), momentum,
                   local_rate);
*/

template <typename Dtype>
void inq_sgd_update_gpu(int N, const Dtype *mask, Dtype *g, Dtype *h,
                        Dtype momentum, Dtype local_rate) {
  INQ_SGDUpdate<Dtype><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, mask, g, h, momentum, local_rate);
  CUDA_POST_KERNEL_CHECK;
}

template void sgd_update_gpu<float>(int, float *, float *, float, float);
template void sgd_update_gpu<double>(int, double *, double *, double, double);
template void inq_sgd_update_gpu<float>(int, const float *, float *, float *,
                                        float, float);
template void inq_sgd_update_gpu<double>(int, const double *, double *,
                                         double *, double, double);

} // namespace caffe
