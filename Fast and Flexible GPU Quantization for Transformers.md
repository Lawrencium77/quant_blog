```toc
```

## Introduction
The goal of quantization is to minimize the precision of model parameters and activations whilst maintaining accuracy. The benefit of this being higher throughput and lower memory footprint.

Disclaimer that this is _scalar_ quantization, not _vector_ quantization.

## Background

We’ll begin with a high-level summary of quantization. If you’d like more reading on this topic, we’ve listed some nice blogs/papers in the References section [1-4].

### The Quantization Equation
In principle, we can use any function to map between floating point (FP) and integer (INT) values. But it’s simplest (and quickest on hardware) to use a linear operation [4]:

$$Q(r)=\textrm{Int}(r/S)-Z$$

where $Q, r$ are the quantized output and floating point input respectively. $S, Z$ are a scale factor and bias. To further minimize inference cost we can set $Z=0$ (we do this for our GPU implementation).

The corresponding dequantization equation is even simpler:

$$\tilde{r}=S(Q(r)+Z$$

Some terminology: Formula (1) describes **uniform quantization** (since the quantized values are uniformly distribute over the input space).

To calculate $S$ we select a **clipping range** $[\alpha, \beta]$ and then do:

$$S=\frac{\beta-\alpha}{2^b-1}$$

where $b$ is the number of bits in our quantization scheme. An extra constraint we choose to enforce is $\alpha=-beta$, which is called **symmetric quantization**. Its benefit is that this is equivalent to setting $Z=0$, which simplifies the (de)quantization function.

### Dynamic vs Static Quantization
A key question is how to determine the clipping range. Too small, and we’ll excessively “truncate” activations & weights. Too big, and we’ll lose precision.

Whilst model parameters can always be quantized offline, its activations can be quantized **dynamically** (the clipping range is calculated for each activation, during a forward pass) or **statically** (also offline). To do static quantization, we run some forward passes and measure the distribution of each activation in the network in order to calculate the clipping range. This process is called **calibration**. For more details, see the next section.

Dynamic quantization is generally more accurate but incurs a computational overhead when calibrating the scalar online. Therefore, **we** **only consider static quantization on GPU** as the scalar reduction is expensive relative to the INT8 matmul, and so can severely limit any performance gains.

### Quantization Granularity
A final distinction to be made is how we share quantization parameters between elements of our parameters and activations. For the rest of this blog, we will use the following diagram to describe a matmul:

![](_attachments/Pasted%20image%2020230313162138.png)

The simplest approach is to use the same scale factor for all elements of W (and likewise for X). This is **per-tensor** quantization.

It’s also feasible to share quantization parameters between some subgroups of each input matrix. A common choice is to assign a specific scale factor to each column of W. This is **per-channel (aka per-column) quantization**.

## Theory
Having covered the basics of quantization, we’ll now look at the important concepts in their implementation.

### Calibration

As explained above, calibration is the process of obtaining activation quantization parameters. We pass a few batches of data through the model to measure the distribution over activations.

There are multiple ways of obtaining a clipping range from these activations. These include:

* Taking a simple min/max
* Minimising KL Divergence between the input and quantized distributions
* Minimising the Mean-Squared Error between input and quantized distributions

We found that the final approach was most performant (although this may be different for different models).

We recommend TensorRT’s PyTorch Quantization Toolkit [5] for the calibration process.

#### Mode 1

#### Mode 2

#### SmoothQuant

* SQ vs INT8.LM()


## Implementation

- Kernel Fusion - Short Description and Link
- Reference Kernel Fusion & Timing Blog

### GPU Quantization in practise

In order to run INT8 GEMMs efficiently on CUDA GPUs we must execute the operation against INT8 Tensor Cores, which were first introduced with the Turing architecture (compute capability 7.0+). This can be achieved by running the `mma.sync.aligned.m8n32k16.row.col.s32.s8.s8.s32` PTX instruction, or calling `wmma::mma_sync` at the CUDA level. However, both approaches require careful management of data movement and layouts to maximize Tensor Core throughput. Thankfully these lower level details are abstracted away by the cuBLASLt  `cublasLtMatmul`  and CUTLASS `device::Gemm`  APIs, both of which support IMMA (integer matrix multiply accumulate).

While they are not currently supported natively in PyTorch, there are other libraries available such as **torch-int** (SmoothQuant) and **bitsandbytes** (LLM.int8()) which expose Python bindings to the underlying C/C++ calls. Microsoft's **ZeroQuant** also leverages CUTLASS but wrappers for their INT8 kernels have not yet been open sourced.

TODO: talk about downside of these libs (i.e. not performant, don't hide overheads) and maybe trim down the following paragraphs

Fully fledged inference frameworks such as NVidia's **TensorRT** or **FasterTransformer** can potentially make things simpler, as they handle the complexity around fusing the quant / dequant to the adjacent operators. This can be a particularly attractive option if you are interested in common Transformer types such as BERT and GPT, for which they have been heavily optimised. However, for anything more exotic it can be a challenge to reach the same levels of performance when factoring in the hard assumptions these libraries make.

When starting from a PyTorch model, TensorRT typically requires an ONNX graph as a starting point, which means all ops must be compatible with the ONNX specification, as well as being compatible with TensorRT _and_ in the required datatype. However this is not the approach taken with TensorRT's flagship BERT implementation which shows an impressive 2x throughput improvement when using INT8 over FP16. Instead, the model is rewritten using the TRT Network Definition API, and utilizes custom plugins (for example fusing the multi-headed attention). This is essential for peak performance, but this level of manual intervention means the benefits of a more generic model export + inference runtime are somewhat diminished. Coupled with the fact that a chunk of TensorRT is a closed source black box, it can lead to a non-trivial development experience.

In contrast **FasterTransformer** is fully open sourced, but still only supports INT8 out of the box with a small number of standard architectures. This is likely because the INT8 compatible models have been rewritten from the ground up in C++, and for good reason: in order to leverage the best performance from cuBLASLt a non-standard interleaved data layout is used for input/output tensors, and so custom activation / normalization kernels are required to avoid expensive layout conversions between matmuls (more on this later...). Whilst this gives performance competitive with TensorRT under certain conditions, it does limit the extensibility of the existing INT8 model implementations for more novel architectures.

As such, for the remainder of the blog we will focus on the challenges of achieving both good INT8 performance _and_ flexibility, by creating modular components that can be applied to different architectures, whilst remaining within the PyTorch framework for the non-quantized components.

### Memory Layouts

As previously suggested, ensuring the input and weight matrices satisfy the memory layout requirements is essential for performance when computing an INT8 GEMM. By default all PyTorch operators expect a **row major** ordering for input and outputs tensors, so ideally we would use the same layout for our INT8 matmul to avoid conversion overhead.

Unfortunately this is not the case with cuBLASLt, which operates on **column major** by default. There is some good news as the `cublasLtMatmul` API supports row major input tensor with column major weight tensor (and we can transpose the weight tensor offline), but the output is returned in column major i.e. input/weight/output = `ROW`/`COL`/`COL`. CUTLASS  goes further and supports `ROW`/`COL`/`ROW` out of the box, which makes it a great option for PyTorch integrations.

While these options are already faster than FP16, to absolutely maximize performance the input tensors must be ordered in the exceptionally non-standard `COL32`/`CUBLASLT_ORDER_COL32_2R_4R4`/`COL32` layout.  `COL32` is an interleaved layout which can be interpreted as row major ordered, but in blocks of 32 columns. CUTLASS supports this by specifying `cutlass::layout::ColumnMajorInterleaved<32>`, where `<1>` would be equivalent to column major and `<n>` where n is equal to the number of columns in the matrix would be equivalent to column major.

`CUBLASLT_ORDER_COL32_2R_4R4` is even more exotic, and is probably best explained visually through the diagrams below which shows a 32x64 matrix, where each value is the address offset in memory for that element.

#### Row major (CUBLASLT_ORDER_ROW)
![](_attachments/Pasted%20image%2020230329111413.png)

#### Column major (CUBLASLT_ORDER_COL)
![](_attachments/Pasted%20image%2020230329111836.png)

#### Column 32 (CUBLASLT_ORDER_COL32)
![](_attachments/Pasted%20image%2020230329111450.png)

#### Column Turing (CUBLASLT_ORDER_COL4_4R2_8C)
![](_attachments/Pasted%20image%2020230329111522.png)

#### Column Ampere (CUBLASLT_ORDER_COL32_2R_4R4)
![](_attachments/Pasted%20image%2020230329111707.png)

By zooming in (4 rows, 16 columns) we hopefully get a clearer picture of the layout pattern

#### Row major (CUBLASLT_ORDER_ROW)
![](_attachments/Pasted%20image%2020230329113112.png)

#### Column major (CUBLASLT_ORDER_COL)
![](_attachments/Pasted%20image%2020230329113209.png)

#### Column 32 (CUBLASLT_ORDER_COL32)
![](_attachments/Pasted%20image%2020230329113410.png)

#### Column Turing (CUBLASLT_ORDER_COL4_4R2_8C)
![](_attachments/Pasted%20image%2020230329113436.png)

#### Column Ampere (CUBLASLT_ORDER_COL32_2R_4R4)
![](_attachments/Pasted%20image%2020230329113458.png)

While `COL32` might be the most performant option, there exists a tension whereby the cost of the layout conversion may cancel out any gains from the reduced precision matmul. Therefore we must either make a design decision à la FasterTransformer and persist the data in the required format, or hide the cost via kernel fusion. The latter approach is similar to how quantize/dequantize overhead is typically hidden, which we will discuss next.


### INT8 GEMM Benchmarking

We now look at some peformance numbers for the various flavours of INT8 GEMM. For these benchmarks we wrap the C++ APIs for cuBLASLt and CUTLASS as PyTorch extensions. A detailed guide to timing CUDA kernels with PyTorch can be found [here](https://www.speechmatics.com/company/articles-and-news/timing-operations-in-pytorch). Benchmarks were run on a T4 GPU with an input tensors of shape [2048, 1920] and [1920, 1920]. Whilst mileage may vary for different input shapes, we found the following conclusions to be consistent over a variety of shapes/sizes.


##### Output precision

<INSERT GEMM DEFINITION>

One important factor which detemines the performance of an INT8 GEMM is the required output type. The matrix multiplication will always have INT8 (i8) dtype for matrices A and B, which then accumulate the outputs into INT32 within the kernel,  but we need to decide whether output matrix C should be INT8 or INT32. 

INT32 return type will be slower as 4 times as much data is written out (and read into the next kernel). We will also have to dequantize after the matmul to return to FP16/FP32. In comparison INT8 return type is faster, but there is a trade-off as accuracy will be impacted as we need to quantize the output from INT32 to INT8 (i.e. requantize) within the kernel. More information on this can be found {earlier in the blog}. If the next operation requires FP16/FP32 we will also have to dequantize, however if we require INT8 for the next kernel an INT8 output type can be ideal. In summary the choice is very much dependant on the accuracy/performance trade-off, as well as the specifics of the model architecture.

|       Kernel       | Time (ms) | vs. FP16 |
|:------------------:|:---------:|:--------:|
| f16f16f16 (Torch)  |    600    |   1.0x   |
| i8i8i32 (cuBLASLt) |    364    |   1.65x  |
| i8i8i8 (cuBLASTLt) |    308    |   1.95x  |






- Torch fp16f16
- CB i8i8
- CB i8i8





i32 vs i8 

i32 vs f16

##### Row major vs. COL32








### Fusion Strategy (and diagrams)

-   Results
	-   Accuracy
	-   [Performance](https://speechmatics.atlassian.net/wiki/spaces/INB/pages/3570565200/Quantization#Performance)
	-   Training Head on top of Body
	-   Sensitivity Analysis

## References
Section 1: Background:

1.     [https://pytorch.org/blog/quantization-in-practice/](https://pytorch.org/blog/quantization-in-practice/)
2.     [https://leimao.github.io/article/Neural-Networks-Quantization/](https://leimao.github.io/article/Neural-Networks-Quantization/)
3.     [https://pytorch.org/docs/stable/quantization.html#model-preparation-for-eager-mode-static-quantization](https://pytorch.org/docs/stable/quantization.html#model-preparation-for-eager-mode-static-quantization)
4.     [https://arxiv.org/pdf/2103.13630.pdf](https://arxiv.org/pdf/2103.13630.pdf)

Section 2: Theory

5.     [https://github.com/NVIDIA/TensorRT/tree/master/tools/pytorch-quantization](https://github.com/NVIDIA/TensorRT/tree/master/tools/pytorch-quantization)
6.

