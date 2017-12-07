# Neural-Network-Model-Compressing

Reproduction &amp; modification of some interesting model compressing methods in Caffe framework, including Dynamic-Network-Surgery (DNS) and Incremental Network Quantization (INQ). Will include more ...

-------
[TOC]
------------

## Dynamic Network Surgery (DNS)

Almost the same with [Guo's version][1].

### DNS usage example

-  Step 1. Change the layer type in the prototxt file as the following table:

   Normal layer type | DNS layer type
   :---: | :---:
   `Convolution` | `DNSConvolution`
   `InnerProduct` | `DNSInnerProduct`

- Step 2. Set DNS layer-specific parameters (see details below)

Below is an example for pruning a `conv` layer.

``` c++
layer {
  name: "conv1"
  type: "DNSConvolution"
  bottom: "data"
  top: "conv1"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: 96
    kernel_size: 11
    stride: 4
    weight_filler {
      type: "gaussian"
      std: 0.01
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
  dns_convolution_param {
    gamma: 0.0000125
    power: 1
    c_rate: -0.7
    iter_stop: 450000
    weight_mask_filler {
      type: "constant"
      value: 1
    }
    bias_mask_filler {
      type: "constant"
      value: 1
    }
  }
}
```

### DNS Notes 
  
  DNS param | Description
  :---: | ---
  `c_rate` | A key parameter for controlling the pruning rate, theoreticall ranging from negative infinity to positive inifinity, but [-2, 5] would be practical. High value yields high pruning rate.
  `gamma` | High value results in low possibility of pruning in each iteration.
  `power` | High value results in low possibility of pruning in each iteration.
  `iter_stop` | The iteration at which the pruning stops, while the training process might continue. If `iter_stop` is set to negative, no pruning will be performed (i.e. the layer will act as a normal `Convolution` or `InnerProduct` layer).
  `weight_mask_filler` & `bias_mask_filler` | must be `"constant"` `"1"`


## Incremental Network Quantization (INQ)

### INQ usage example

-  Step 1. Change the layer type in the prototxt file as the following table:

   Normal layer type | DNS layer type
   :---: | :---:
   `Convolution` | `INQConvolution`
   `InnerProduct` | `INQInnerProduct`

- Step 2. Set INQ layer-specific parameters (see details below)

Below is an example for quantizing an `ip` layer.

``` c++
layer {
  name: "ip1"
  type: "INQInnerProduct"
  bottom: "pool2"
  top: "ip1"
  param {
    lr_mult: 1
  }
  param {
    lr_mult: 2
  }
  inner_product_param {
    num_output: 500
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
    }
  }
  inq_inner_product_param {
    portion: 0.6
    portion: 0.8
    num_quantum_values: 7
    weight_mask_filler {
      type: "constant"
      value: 1
    }
    bias_mask_filler {
      type: "constant"
      value: 1
    }
  }
}
```


### INQ Notes 
  
  Since INQ requires to quantize the parameters 

  INQ param | Description
  :---: | ---
  `portion` | the portions of quantized weights, see more details at below.
  `num_quantum_values` | The number of exponent to be kept, see more details below.
  `weight_mask_filler` & `bias_mask_filler` | must be `"constant"` `"1"`

- **More details**
`portion`: assuming that we set the accumulated
portions of quantized weights at iterative steps as {0.3, 0.6, 0.8, 1}, the INQ parameter settings in the above example therefore denotes that the current training will make the quantized weights from 60% to 80%. 
**_More about_** **`portion`**: Actually, 





[1]: https://github.com/yiwenguo/Dynamic-Network-Surgery


