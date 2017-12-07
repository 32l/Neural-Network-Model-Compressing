/*
the same with DNS convolution code
*/

#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/cconv_layer.hpp"

namespace caffe {

template <typename Dtype>
void CConvolutionLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  BaseConvolutionLayer <Dtype>::LayerSetUp(bottom, top); 
  
  /********** for neural network model compression **********/
  CConvolutionParameter dsn_conv_param = this->layer_param_.cconvolution_param();
  
  if(this->blobs_.size()==2 && (this->bias_term_)){
    this->blobs_.resize(4);
    // Intialize and fill the weight mask & bias mask
    this->blobs_[2].reset(new Blob<Dtype>(this->blobs_[0]->shape()));
    shared_ptr<Filler<Dtype> > weight_mask_filler(GetFiller<Dtype>(
        dsn_conv_param.weight_mask_filler()));
    weight_mask_filler->Fill(this->blobs_[2].get());
    this->blobs_[3].reset(new Blob<Dtype>(this->blobs_[1]->shape()));
    shared_ptr<Filler<Dtype> > bias_mask_filler(GetFiller<Dtype>(
        dsn_conv_param.bias_mask_filler()));
    bias_mask_filler->Fill(this->blobs_[3].get());    
  }  
  else if(this->blobs_.size()==1 && (!this->bias_term_)){
    this->blobs_.resize(2);   
    // Intialize and fill the weight mask
    this->blobs_[1].reset(new Blob<Dtype>(this->blobs_[0]->shape()));
    shared_ptr<Filler<Dtype> > bias_mask_filler(GetFiller<Dtype>(
        dsn_conv_param.bias_mask_filler()));
    bias_mask_filler->Fill(this->blobs_[1].get());      
  }  
  
  // Intializing the tmp tensor
  this->weight_tmp_.Reshape(this->blobs_[0]->shape());
  this->bias_tmp_.Reshape(this->blobs_[1]->shape());  
  
  // Intialize the hyper-parameters
  this->std_ = 0;
  this->mu_ = 0;   
  this->gamma_ = dsn_conv_param.gamma(); 
  this->power_ = dsn_conv_param.power();
  this->c_rate_ = dsn_conv_param.c_rate();  
  this->iter_stop_ = dsn_conv_param.iter_stop();
  /**********************************************************/
}

template <typename Dtype>
void CConvolutionLayer<Dtype>::compute_output_shape() {
  const int* kernel_shape_data = this->kernel_shape_.cpu_data();
  const int* stride_data = this->stride_.cpu_data();
  const int* pad_data = this->pad_.cpu_data();
  const int* dilation_data = this->dilation_.cpu_data();
  this->output_shape_.clear();
  for (int i = 0; i < this->num_spatial_axes_; ++i) {
    // i + 1 to skip channel axis
    const int input_dim = this->input_shape(i + 1);
    const int kernel_extent = dilation_data[i] * (kernel_shape_data[i] - 1) + 1;
    const int output_dim = (input_dim + 2 * pad_data[i] - kernel_extent)
        / stride_data[i] + 1;
    this->output_shape_.push_back(output_dim);
  }
}

template <typename Dtype>
void CConvolutionLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {  
      
  const Dtype* weight = this->blobs_[0]->cpu_data();    
  Dtype* weightMask = this->blobs_[2]->mutable_cpu_data(); 
  Dtype* weightTmp = this->weight_tmp_.mutable_cpu_data(); 
  const Dtype* bias = NULL;
  Dtype* biasMask = NULL;  
  Dtype* biasTmp = NULL;
  if (this->bias_term_) {
    bias = this->blobs_[1]->cpu_data(); 
    biasMask = this->blobs_[3]->mutable_cpu_data();
    biasTmp = this->bias_tmp_.mutable_cpu_data();
  }

  if (this->phase_ == TRAIN){
    // Calculate the mean and standard deviation of learnable parameters 
    if (this->std_==0 && this->iter_ ==0){
      Dtype sum_mu = 0, sum_square = 0;
      unsigned int nz_w = 0, nz_b = 0, ncount = 0;
      for (unsigned int k = 0;k < this->blobs_[0]->count(); ++k) {
        Dtype mtw = weightMask[k]*weight[k];
        sum_mu += fabs(mtw);
        sum_square += mtw*weight[k];
        if (mtw!=0) nz_w++;
      }
      if (this->bias_term_) {
        for (unsigned int k = 0;k < this->blobs_[1]->count(); ++k) {
          Dtype mtb = biasMask[k]*bias[k];
          sum_mu += fabs(mtb);
          sum_square += mtb * bias[k];
          if (mtb!=0) nz_b++;
        }       
      }
      ncount = nz_w + nz_b;
      this->mu_ = sum_mu / ncount;
      this->std_ = sqrt((sum_square - ncount * sum_mu*sum_mu)/ncount);
      // output the percentage of kept parameters.
      LOG(INFO)<< "mu_:" <<mu_<<" "<<"std_:"<<std_<<" "
               << nz_w <<"/" <<this->blobs_[0]->count() 
               << "(" << Dtype(nz_w)/this->blobs_[0]->count() << ")" << " "
               << nz_b <<"/" <<this->blobs_[1]->count() 
               << "(" << Dtype(nz_b)/this->blobs_[1]->count() << ")" << " "
               << ncount<<"/" << this->blobs_[0]->count()+this->blobs_[1]->count()
               << "(" <<Dtype(ncount)/(this->blobs_[0]->count()+this->blobs_[1]->count()) << ")"
               << "\n";
    }
    // Calculate the weight mask and bias mask with probability
    Dtype r_ = static_cast<Dtype>(rand())/static_cast<Dtype>(RAND_MAX);
    if (pow(1+(this->gamma_)*(this->iter_),-(this->power_))>r_ && (this->iter_)<(this->iter_stop_)) {  
      for (unsigned int k = 0;k < this->blobs_[0]->count(); ++k) {
        if (weightMask[k]==1 && fabs(weight[k]) <= 0.9 * std::max(mu_+c_rate_*std_,Dtype(0))) 
          weightMask[k] = 0;
        else if (weightMask[k]==0 && fabs(weight[k])>1.1*std::max(mu_+c_rate_*std_,Dtype(0)))
          weightMask[k] = 1;
      } 
      if (this->bias_term_) {       
        for (unsigned int k = 0;k < this->blobs_[1]->count(); ++k) {
          if (biasMask[k]==1 && fabs(bias[k])<=0.9*std::max(mu_+c_rate_*std_,Dtype(0))) 
            biasMask[k] = 0;
          else if (biasMask[k]==0 && fabs(bias[k])>1.1*std::max(mu_+c_rate_*std_,Dtype(0)))
            biasMask[k] = 1;
        } 
      } 
    }
  } 

  // Calculate the actual forwarded (masked) weight and bias
  for (unsigned int k = 0;k < this->blobs_[0]->count(); ++k) {
    weightTmp[k] = weight[k]*weightMask[k];
  }
  if (this->bias_term_){
    for (unsigned int k = 0;k < this->blobs_[1]->count(); ++k) {
      biasTmp[k] = bias[k]*biasMask[k];
    }
  }
  
  // Forward calculation with masked weight and bias 
  for (int i = 0; i < bottom.size(); ++i) {
    const Dtype* bottom_data = bottom[i]->cpu_data();
    Dtype* top_data = top[i]->mutable_cpu_data();
    for (int n = 0; n < this->num_; ++n) {
      this->forward_cpu_gemm(bottom_data + n * this->bottom_dim_, weightTmp,
          top_data + n * this->bottom_dim_);
      if (this->bias_term_) {
        this->forward_cpu_bias(top_data + n * this->top_dim_, biasTmp);
      }
    }
  }
}

template <typename Dtype>
void CConvolutionLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* weightTmp = this->weight_tmp_.cpu_data();  
  const Dtype* weightMask = this->blobs_[2]->cpu_data();
  Dtype* weight_diff = this->blobs_[0]->mutable_cpu_diff();
  for (int i = 0; i < top.size(); ++i) {
    const Dtype* top_diff = top[i]->cpu_diff();    
    // Bias gradient, if necessary.
    if (this->bias_term_ && this->param_propagate_down_[1]) {
      const Dtype* biasMask = this->blobs_[3]->cpu_data();
      Dtype* bias_diff = this->blobs_[1]->mutable_cpu_diff();    
      for (unsigned int k = 0;k < this->blobs_[1]->count(); ++k) {
        bias_diff[k] = bias_diff[k]*biasMask[k];
      }
      for (int n = 0; n < this->num_; ++n) {
        this->backward_cpu_bias(bias_diff, top_diff + n * this->top_dim_);
      }
    }
    if (this->param_propagate_down_[0] || propagate_down[i]) {
      const Dtype* bottom_data = bottom[i]->cpu_data();
      Dtype* bottom_diff = bottom[i]->mutable_cpu_diff(); 
      for (unsigned int k = 0;k < this->blobs_[0]->count(); ++k) {
        weight_diff[k] = weight_diff[k]*weightMask[k];
      }
      for (int n = 0; n < this->num_; ++n) {
        // gradient w.r.t. weight. Note that we will accumulate diffs.
        if (this->param_propagate_down_[0]) {
          this->weight_cpu_gemm(bottom_data + n * this->bottom_dim_,
              top_diff + n * this->top_dim_, weight_diff);
        }
        // gradient w.r.t. bottom data, if necessary.
        if (propagate_down[i]) {
          this->backward_cpu_gemm(top_diff + n * this->top_dim_, weightTmp,
              bottom_diff + n * this->bottom_dim_);
        }
      }
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(CConvolutionLayer);
#endif

INSTANTIATE_CLASS(CConvolutionLayer);
REGISTER_LAYER_CLASS(CConvolution);

}  // namespace caffe
