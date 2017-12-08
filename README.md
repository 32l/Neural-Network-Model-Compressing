# Neural-Network-Model-Compressing

Reproduction &amp; modification of some interesting model compressing methods in Caffe framework, including Dynamic-Network-Surgery (DNS) and Incremental Network Quantization (INQ). 

The code in this repo is based on the master branch of [BVLC/caffe][2] (2017/08/28)

**Features:**

- DNS fine-tuning (pruning)
- INQ fine-tuning (quantization)
- Support warm-up training
- Python scripts for Caffe model checking / model conversion / model comprssion


[TOC]
------------


## 1 Dynamic Network Surgery (DNS)

Almost the same with [Guo's version][1].

### 1.1 DNS usage example

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

### 1.2 DNS Notes 
  
  DNS param | Description
  :---: | ---
  `c_rate` | A key parameter for controlling the pruning rate, theoreticall ranging from negative infinity to positive inifinity, but [-2, 5] would be practical. High value yields high pruning rate.
  `gamma` | High value results in low possibility of pruning in each iteration.
  `power` | High value results in low possibility of pruning in each iteration.
  `iter_stop` | The iteration at which the pruning stops, while the training process might continue. If `iter_stop` is set to negative, no pruning will be performed (i.e. the layer will act as a normal `Convolution` or `InnerProduct` layer).
  `weight_mask_filler` & `bias_mask_filler` | must be `"constant"` `"1"`


## 2 Incremental Network Quantization (INQ)

### 2.1 INQ usage example

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


### 2.2 INQ Notes 
  
  Since INQ requires to quantize the parameters 

  INQ param | Description
  :---: | ---
  `portion` | the portions of quantized weights, must be set, only the first two will be used. see more details at below.
  `num_quantum_values` | The number of the exponent of 2 to be kept, see more details below.
  `weight_mask_filler` & `bias_mask_filler` | must be `"constant"` `"1"`

- **More details**
  - **`portion`**
    assuming that we set the accumulated portions of quantized weights at iterative steps as {0.3, 0.6, 0.8, 1}, the INQ parameter settings in the above example therefore denotes that the current training will increase the portion of quantized weights in `ip1` layer from 60% to 80%. 
    **_More about_** **`portion`**: Actually, the `portion` pair only     works on non-zero weights, which is especially useful when    applying INQ to a DNS-pruned model, freeing you from concerning   about the actual pruning rate to get the portion pair correctly.
    For example, if a layer contains 100 weights (we are omitting the     bias), 90% pruned away after DNS (i.e. 90 weights of 0), 10     weights remaining (i.e. 10 weights of non-zero), the following    `portion` pair will quantize only 3 weights, rather than 30   weights:
    ```
    {
      portion: 0.0
      portion: 0.3
    }
    ```
    and the following `portion` pair will quantize only 2 weights:
    ```
    {
      portion: 0.5
      portion: 0.6
    }
    ```
  - `num_quantum_values`
    The number of exponent of 2 to be ketp. For example, if `num_quantum_values` is set to 7, the weights in a layer after quantization could fall in the set of {-2^-7, -2^-6, ..., -2^-1, 0, 2^-7, 2^-6, ..., 2^-2, 2^-1}, 7 exponents (-1, -2, ..., -7) kept. The layer's exponent set could be {-2, -3, ..., -8}, {0, -1, ..., -6} ..., depending on the weights of largest absolute value of the layer.
>

- **Important**
  If you would like to apply INQ to a DNS-pruned model, a model conversion must be done before you run the INQ fine-tuning. Once you get the raw INQ model, it safe to start INQ fine-tuning on that model. See more details about the conversion scripts `model2INQ_raw.py`.


## 3 Python Scripts

---
**Note**

- All the python scripts locate in the folder `compression_scripts`, and the usage of all scripts can be checked by running :
  ``` shell
  python script_name.py
  ```
  In case I did not write any description about a script, you can   do the above.
>

- Make sure you have set **`CAFFE_ROOT`** environment variable in you system (set the `CAFFE_ROOT` as your caffe directory), or just type 
  ``` shell
  export CAFFE_ROOT=dir/to/your/caffe_dir
  ```
  every time you want to use these scripts.
>

- Make sure you have compiled pycaffe by running the following command:
  ``` shell
  make pycaffe
  ```

---

### 3.1 Model Conversion Scripts

- **`dns_to_normal.py`**

  > This script is used for converting the raw DNS caffemodel (output caffemodel of DNS pruning) to normal caffemodel (about half size of the raw DNS caffemodel) as well as checking the compression rate for each learnable layer.

  After compiling the caffe and pycaffe, prepare your compressed  DNS caffemodel, and run the following command from your  CAFFE_ROOT (make sure you have set `CAFFE_ROOT` environment  variable, which is the dir of you caffe folder) :

  ``` shell
  python compression_scripts/dns_to_normal.py <dns.prototxt>  <dns_model.caffemodel> <target.prototxt>   <output_target.caffemodel>
  ```

  e.g.

  ``` shell
  python compression_scripts/dns_to_normal.py   examples/mnist/dns_train_val.prototxt   examples/mnist/dns_iter_10000.caffemodel  examples/mnist/mnist_train_val.prototxt  examples/mnist/mnist_test_DNS.caffemodel
  ```


  After running the above command, the compression rate should be   shown on the screen, and the `output_target.caffemodel` should  have the same size as a normal caffemodel (about 1/2 of the  dns_model.caffemodel) which can be used for testing with normal  prototxt file.

  E.g.
  ``` shell
  ./build/tools/caffe test  -model=examples/mnist/lenet_train_test.prototxt  -weights=examples/mnist/mnist_test_DNS.caffemodel -gpu=all
  ```

  `dns_to_normal.py` is compatible with Yiwen Guo' DNS raw  caffemodel (which is twice the size of the normal model).
>


- **`inq_to_normal.py`**

  > This script is used for converting the raw INQ caffemodel (output caffemodel of INQ quantization) to normal caffemodel (about half size of the raw INQ caffemodel)

  Usage is similar to `dns_to_normal.py`
>

- **`model2INQ_raw.py`**
  > This script is used for converting raw DNS model or normal model to raw INQ model. This script must be used if you want to apply INQ to a normal model or DNS model since the INQ layer will not recognize them.

  Run following command from your `CAFFE_ROOT` to see usage. Yes, I am being lazy...
  ``` shell
  python compression_scripts/model2INQ_raw.py
  ```

### 3.2 Model Checking Scripts

- `parse_model.py`
  > This script is used for converting binary caffemodel (*.caffemodel) to txt file, so that you can read the actual data in caffemodel. Can be used for any caffe model (the normal ones, DNS ones, INQ ones, ...)

  Usage:
  ``` shell
  python compression_scripts/parse_model.py <binary.caffemodel> <output.txt>
  ```



### 3.3 Model Compression Scripts

 - this part is usually highly customized since you may want to use different format to store your model.

to be continued...



TODO

more description of scripts to add ...

----



[1]: https://github.com/yiwenguo/Dynamic-Network-Surgery
[2]: https://github.com/BVLC/caffe


