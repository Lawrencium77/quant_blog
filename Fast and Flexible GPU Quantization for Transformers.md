## Introduction
As transformer models increase in size, the computational cost of running inference also grows. Many companies now face the challenge of deploying state-of-the-art models in a cost-effective way.

This has led to a surge in interest for optimizing transformer inference. There are a range of techniques available [1], including:

* Sparsity
* Pruning
* Conditional Computation (e.g., Mixture of Experts)
* Knowledge Distillation
* Quantization

Of these, quantization is in some sense the most universal. It can be applied to any network, regardless of architecture.

By reducing the precision of model parameters and activations, quantization aims to minimize increase throughput and decrease memory footprint, at the cost of potentially damaging model accuracy.

Provided the decrease in accuracy is minimal, this sounds ideal. However, implementing a GPU-based quantization scheme that *actually speeds up your model* is not straightforward. The main challenges stem from a lack of fast General Matrix Multiply (GEMM) GPU kernels for precisions lower than INT8, and the overheads associated with converting between floating point (FP) and Integer (INT) data types.

In this blog, we provide a detailed guide to GPU-based quantization of transformers. We describe an approach that is both flexible and capable of genuinely improve throughput. The content is organized as follows:

* [Background](#Background)
	* [The Quantization Equation](#The%20Quantization%20Equation)
	* [Dynamic vs Static Quantization](#Dynamic%20vs%20Static%20Quantization)
	* [Quantization Granularity](#Quantization%20Granularity)
* [Important Concepts](#Important%20Concepts)
	* [Calibration](#Calibration)
	* [Specifics of INT8 GEMMs](#Specifics%20of%20INT8%20GEMMs)
		* [i8f16](#i8f16)
		* [i8i8](#i8i8)
		* [Quantization Operation Overheads](#Quantization%20Operation%20Overheads)
	* [SmoothQuant](#SmoothQuant)
* [Implementation](#Implementation)
	* [GPU Quantization in Practice](#GPU%20Quantization%20in%20Practice)
	* [Memory Layouts](#Memory%20Layouts)
	* [Fusion Strategy (and diagrams)](#Fusion%20Strategy%20(and%20diagrams))
* [Some Brief Results](#Some%20Brief%20Results)
	* [Accuracy](#Accuracy)
	* [Throughput](#%20Throughput)
* [References](#References)



## Background

We’ll begin with a quick summary of quantization. For further reading on this subject, we’ve listed some nice blogs/papers in our [References](#References) section [1-4].

### The Quantization Equation
In principle, we can use any function to convert from a higher-precision to lower-precision representation. But a linear function is simplest and quickest on the hardware [4]:

$$Q(r)=\textrm{Int}(r/S)-Z$$

Here, $Q, r$ are the INT output and FP input, while $S, Z$ are a scale factor and bias. $\textrm{Int}$ is a function that rounds to the nearest integer. To minimize inference cost, we can set $Z=0$ (we do this for our GPU implementation). 

The corresponding dequantization equation is even simpler:

$$\tilde{r}=S Q(r)$$

This method is called **uniform quantization** since the quantized values are uniformly distributed over the input space. To calculate $S$ we select a **clipping range** $[\alpha, \beta]$ and then use:

$$S=\frac{\beta-\alpha}{2^b-1}$$

Here, $b$ is the number of bits in our quantization scheme. We choose to enforce $\alpha=-\beta$, which is known as **symmetric quantization**. This simplifies the (de)quantization function by setting $Z=0$.

It's important to note that the rounding function in Equation (1) incurs a loss of information. In general, $\tilde{r}=SQ(r)\not = r$.  The value $\tilde{r}-r$ is called **quantization error**.

### Dynamic vs Static Quantization
A key question is how to determine the clipping range. Too small, and we’ll excessively “truncate” activations and weights. Too big, and we’ll lose precision.

While model parameters can always be quantized offline, its activations can be quantized **dynamically** (with the clipping range calculated for each activation during a forward pass) or **statically** (also offline). Static quantization involves running some forward passes, measuring the distribution of each activation in the network, and calculating the clipping range. This process is called **calibration**. For more information, see the next section.

Dynamic quantization tends to be more accurate but requires additional computational overhead for online scalar calibration. As a result, **we only consider static quantization on GPU** because scalar reduction (relative to an INT8 matmul) can be costly and limit performance gains.

### Quantization Granularity
A final distinction to be made is how we quantization parameters are shared between elements of our parameters and activations. Throughout this blog, we'll use the following diagram to illustrate a matmul:

![](_attachments/Pasted%20image%2020230313162138.png)

The simplest approach is to use the same scale factor for all elements of $W$ (and likewise for $X$). This is known as **per-tensor** quantization.

It’s also feasible to share quantization parameters between some subgroups of each input matrix. A popular option is to assign a specific scale factor to each column of $W$, referred to as **per-channel (or per-column) quantization**. This is more accurate than per-tensor quantization; using a specific scale means the error incurred in quantizing each column is lower. 

## Important Concepts
With the fundamentals of quantization covered, let's explore the important concepts in its implementation.

### Calibration

Calibration, as mentioned above, involves obtaining activation quantization parameters by passing several batches of data through the model to measure activation distribution.

There are multiple methods to derive a clipping range from these activations, such as:

* Using a simple min/max
* Minimising KL Divergence between the input and quantized distributions
* Minimising the Mean-Squared Error between input and quantized distributions

We found that the final approach was most performant (although this may vary for different models).

To perform the calibration process, we recommend using TensorRT’s PyTorch Quantization Toolkit [5]. 

### Specifics of  INT8 GEMMs
The core element of a quantized neural network is INT8 matrix multiplication. Focusing on its details is crucial for an efficient implementation. This section describes these details, and serves as context for the later section describing [Implementation](#Implementation).

We identify two types of INT8 matmul, differentiated by their return type. We'll discuss each of these in turn.

#### i8f16
Consider the following matrix multiplication:

$$Y=WX+b$$

where $X\in \mathbb{R}^{N \times d}$, $W\in \mathbb{R}^{d \times d}$, $Y\in \mathbb{R}^{N \times d}$, $b\in \mathbb{R}^{d}$  are the input, weight, output, and bias tensors respectively. Consider the case where all tensors are **Floating Point**, but the matrix multiply itself runs in INT8. An example INT8 to FP16 (i8f16) matrix multiplication would be implemented as follows:

![](_attachments/Mode%201%20GEMM%20(1).svg)

There are several points to note:

* The input $X$ first passes through a quantization operation, labelled Q. This performs the operation described in Equation (1).
* Our weights $W$ can be quantized offline. 
* The output of the Matmul has **INT32** dtype. The structure used to contain this output is called the **accumulator**. The accumulator value is passed through a dequantization op, labelled DQ. This performs the operation described in Equation (2).
* The bias step is not quantized.

Multiplication of two signed INT8 numbers can be represented by in INT16. Since a matmul involves the addition of several INT16 values, the accumulator must have dtype INT32 to prevent overflow.

#### i8i8
Returning in INT8 involves an extra step:

![](_attachments/Blank%20diagram%20(1).svg)

In this **requantization** step, labelled RQ, we convert the INT32 representation back into INT8. The benefit is a reduction in the amount of data written from GPU SRAM to DRAM.

#### Quantization Operation Overheads
To fully realise the throughput improvements from INT8 matrix multiplications, we must mitigate the cost of the Q/DQ/RQ nodes. Since these are elementwise operations, this can be achieved through [operator fusion](https://horace.io/brrr_intro.html). 
The following diagrams demonstrate this for i8f16 and i8i8. Fused operators are indicated by the dashed boxes:

![](_attachments/Mode%201%20GEMM%20(2).svg)

![](_attachments/Blank%20diagram%20(2).svg)

In both cases, the Q node can sometimes be fused with a preceding operation (usually a layernorm). 
In i8f16, we see the DQ is fused with the matrix multiply itself. This ensures the dtype of the tensor that's transferred between SRAM and DRAM is FP16 instead of INT32.
In i8i8, we see the RQ is fused with the matmul. This ensures an INT8 return type. The DQ is fused with the bias add, as well as any ops that might follow (for example, a residual add).


### SmoothQuant
In this section, we give an intuition behind SmoothQuant - a recent paper that addresses accuracy degradation when quantizing neural nets. We found this to be surprisingly effective for our own models. Importantly, SmoothQuant can be applied **offline**, meaning there are no downsides related to throughput or memory footprint.

The authors describe two key observations that motivate SmoothQuant:

1. The distribution of neural network weights is uniform and flat. The distributions of activations is not. This makes activations harder to quantize than weights.
2. Activation outliers appear in fixed channels.

The following diagram, taken from the original paper, illustrates these ideas for a single linear layer:

![](_attachments/Screenshot%202023-03-29%20at%2014.43.18.png)

On the left-hand side, we see dramatic outlier channels in the input tensor. Given this, an obvious solution would be to apply a per-channel quantization factor. Unfortunately, this is not feasible: applying a scaling factor to individual columns of the input tensor would not factor out nicely in the output, meaning we could not apply dequantization.

Other works have instead used a per-token quantization granularity. This can improve accuracy slightly, but does not solve the issue of fixed-channel outlier values. 

Instead, SmoothQuant aims to "migrate" the quantization difficulty from activations to weights. It does so by scaling each channel of the activations by a "smoothing factor". To ensure mathematical equivalence, we must scale each token of the weight tensor by the same amount in the opposite direction.

Mathematically, this is given by:

$$Y = (X\textrm{diag}(s)^{-1})\cdot(\textrm{diag}(s)W)=\hat{X}\hat{W}$$

where $s\in \mathbb{R}^d$  is our smoothing factor. Here's a diagram, again taken from the paper:

![](_attachments/Screenshot%202023-03-29%20at%2015.01.32.png)

All that remains is how to determine $s$. Since quantization is easiest when all channels have the same maximum value, one possibility is:

$$s_j=\max(|X_j|)$$
where $j$ is the channel index. This ensures that all channels would have the same maximum value (of 1). However, this may push too much of the quantization difficulty to the weights, meaning we harm quantization accuracy.

The other extreme is:

$$s_j = 1 / \max({|W_j|})$$
To control the migration strength, the authors propose combining each of these equations by introducing a hyperparameter, $\alpha \in [0,1]$:

$$s_j=\frac{\max(|X_j|)^\alpha}{\max({|W_j|})^{1-\alpha}}$$

$\alpha=1$ corresponds to migrating all difficulty to the weights. $\alpha=0$ migrates all difficulty to the activations. In general we found setting $\alpha$ to be between 0.5 and 0.9 achieved good performance.

It's important to appreciate that this smoothing process can be applied **offline**. For the weights, this is trivial. For the activations, we exploit the fact that GEMM operations in a transformer block often follow a layernorm. Combining the multiplication by $\textrm{diag}(s)^{-1}$  into the layernorm parameters means that it too can be done offline.
A consequence of this is that SmoothQuant can only be applied to matrix multiplications that follow an operation which, like Layernorm, can accommodate any smoothing factor into its parameters. The diagram below indicates the relevant matrix multiplies in a standard transformer block:

![](_attachments/Blank%20diagram%20(4).svg)


## Implementation

- Kernel Fusion - Short Description and Link
- Reference Kernel Fusion & Timing Blog

### GPU Quantization in Practice

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

We now look at peformance numbers for the various flavours of INT8 GEMM. For these benchmarks we wrap the C++ APIs for cuBLASLt and CUTLASS as PyTorch extensions. A detailed guide to timing CUDA kernels with PyTorch can be found [here](https://www.speechmatics.com/company/articles-and-news/timing-operations-in-pytorch). Benchmarks were run on a T4 GPU with input tensors of shape [2048, 1920] and [1920, 1920]. While mileage may vary for different input shapes, we found the following conclusions to be consistent over a variety of shapes/sizes.

####  INT8 vs INT32 output precision

$$D=\alpha AB+\beta C$$

One important factor which determines INT8 GEMM performance (formula above) is the required output type. The matrix multiplication will always have INT dtype for matrices A and B, which then accumulate the outputs into INT32 within the kernel,  but we need to decide whether output C should be INT8 or INT32. 

INT32 return type will be slower as four times as much data is written out (and read into the next kernel). We will also have to dequantize after the matmul to return to FP16. In comparison, INT8 return type is faster but there is a trade-off: accuracy will be made worse, as we need to requantize the output from INT32 to INT8 within the kernel. More information on this can be found {earlier in the blog}. If the next operation requires FP16 input we will also have to dequantize. However, if we require INT8 for the next kernel an INT8 output type can be ideal.

In summary, the decision is very much dependent on the accuracy/performance trade-off, as well as the specifics of the model architecture.

|       Kernel       | Time (ms) | vs. FP16 |
|:------------------:|:---------:|:--------:|
| f16f16f16 (Torch)  |    600    |   1.0x   |
| i8i8i32 (cuBLASLt) |    364    |   1.65x  |
| i8i8i8 (cuBLASLt) |    308    |   1.95x  |

#### FP16 output precision

We previously touched upon the fact that INT32 return type requires dequantizing outside of the matmul. Performance could be improved by fusing the dequant with the matmul and returning FP16 outputs. We can get this behaviour for free by using the GEMM `α` parameter to dequantize the outputs (the same way that we requantize for INT8 outputs), but this only works if we are applying **per-tensor** quantization, where the dequantization parameter is a single scalar {refer to per-scalar/per-channel section for more detail}.

What if we require **per-channel** quantization i.e. using a vector to dequantize? In this scenario CUTLASS comes to the rescue by allowing the definition of a custom epilogue function, which is applied after the matrix multiplication, in a single fused kernel. For this scenario the GEMM + epilogue definition is expanded to 

$$D=f_2(f_1(\alpha AB+\beta C, d)) $$
The epilogue format comes from `EpilogueWithBroadcast` which applies a [binary operation](https://github.com/NVIDIA/cutlass/blob/master/include/cutlass/epilogue/thread/linear_combination_bias_elementwise.h#L215)`f1` between the matmul output and a column-wise broadcasted vector `d`, followed by an optional elementwise op `f2`.

This might typically be a bias addition followed by an activation function (e.g. ReLU), but in our case we want `f1` to be a multiplication with the dequantization scalar. The epilogue is then plugged into `gemm::device::GemmUniversalWithBroadcast`.

|       Kernel       | Time (ms) | vs. FP16 |
|:------------------:|:---------:|:--------:|
| f16f16f16 (Torch)  |    600    |   1.0x   |
| i8i8i32 (CUTLASS) |    364    |   1.65x  |
| i8i8f16 (CUTLASS) |    308    |   1.95x  |

While there might not be a huge improvement from FP16 output in terms of GEMM throughput, there are other peformance benefits:
- 50% less data loaded in the next kernel (now FP16 instead of INT32)
- Avoid fusion of the dequantize operator with the next kernel
- Avoid loading the dequantization vector in the next kernel (which CUTLASS pipelines the loading of TODO improve this sentence)


#### Memory layout

Lastly we examine the effect of the aforementioned layout in memory on the matmul performance 

|       Kernel       | Time (ms) | vs. FP16 |
|:------------------:|:---------:|:--------:|
| FP16 Row major (Torch)  |    600    |   1.0x   |
| INT8 Row major  (cuBLASLt) |    365    |   1.64x  |
| INT8 COL32 (cuBLASLt) |    308    |   1.95x  |









### Fusion Strategy (and diagrams)

## Some Brief Results
### Accuracy
### Throughput

## References
Section 0: Intro
1. https://lilianweng.github.io/posts/2023-01-10-inference-optimization/

Section 1: Background:

1.     [https://pytorch.org/blog/quantization-in-practice/](https://pytorch.org/blog/quantization-in-practice/)
2.     [https://leimao.github.io/article/Neural-Networks-Quantization/](https://leimao.github.io/article/Neural-Networks-Quantization/)
3.     [https://pytorch.org/docs/stable/quantization.html#model-preparation-for-eager-mode-static-quantization](https://pytorch.org/docs/stable/quantization.html#model-preparation-for-eager-mode-static-quantization)
4.     [https://arxiv.org/pdf/2103.13630.pdf](https://arxiv.org/pdf/2103.13630.pdf)

Section 2: Theory
5. [https://github.com/NVIDIA/TensorRT/tree/master/tools/pytorch-quantization](https://github.com/NVIDIA/TensorRT/tree/master/tools/pytorch-quantization)
6. https://arxiv.org/abs/2211.10438
7. 
