---
layout: posts
title: "使用NVIDIA Nsight对llamacpp进行性能分析"
subtitle: ""
description: "Nsight使用示例"
excerpt: ""
date: 2025-07-14 12:00:00
author: "rickyang"
image: "/images/posts/8.jpg"
published: true
tags:
  - llamacpp
  - cuda
URL: "/2025/07/14/nsys-analysis-llamacpp"
categories:
  - mlsys
  - AI
is_recommend: true
---

# 使用NVIDIA Nsight Systems对llamacpp进行性能分析

## 环境准备

在开始之前，请确保您已准备以下内容：

- 已安装 NVIDIA Nsight Systems
- 一块兼容的 NVIDIA GPU
- llamacpp 可执行文件以及所需的模型文件（例如 `gemma-3-4b-it-f16.gguf`）

## 分析步骤

### 步骤 1：指定 GPU 设备

通过设置 `CUDA_VISIBLE_DEVICES` 环境变量来指定用于性能分析的 GPU 设备。

```bash
export CUDA_VISIBLE_DEVICES=1
```

### 步骤 2：启动 llamacpp 服务器并启用性能分析

使用 `nsys profile` 命令启动 llamacpp 服务器并启用性能分析。以下命令指定了性能分析数据的输出文件以及服务器运行的参数：

```bash
nsys profile -o dev1-gemma-4b.nsys-rep bin/llama-server -ngl 81 -t 8 -c 0 --port 8000 -fa -m /media/do/sata-512G/modelhub/gemma-3-4b-it-f16.gguf
```

- `-o dev1-gemma-4b.nsys-rep`：指定性能分析报告的输出文件名。
- `bin/llama-server`：llamacpp 服务器的可执行文件。
- `-ngl 81`：指定 GPU 层数。
- `-t 8`：设置线程数。
- `-c 0`：设置上下文大小。
- `--port 8000`：指定服务器端口。
- `-fa`：启用快速注意力机制。
- `-m /media/do/sata-512G/modelhub/gemma-3-4b-it-f16.gguf`：模型文件的路径。

启动服务器后，向其发送推理请求以生成性能分析数据。

### 步骤 3：安全关闭 llamacpp 服务器

在收集到足够的性能数据后，使用以下命令安全关闭 llamacpp 服务器：

```bash
pidof llama-server | xargs kill
```

此命令会查找 llamacpp 服务器的进程 ID 并将其优雅地终止。

### 步骤 4：分析性能数据

使用 `nsys stats` 命令分析生成的性能数据文件：

```shell
nsys stats dev1-gemma-4b.nsys-rep
```

```
 ** CUDA API Summary (cuda_api_sum):

 Time (%)  Total Time (ns)  Num Calls    Avg (ns)      Med (ns)    Min (ns)    Max (ns)    StdDev (ns)                 Name               
 --------  ---------------  ---------  ------------  ------------  ---------  -----------  ------------  ---------------------------------
     81.2   18,784,449,306     13,185   1,424,683.3       6,991.0        306   26,541,226   4,702,507.3  cudaStreamSynchronize            
     10.0    2,306,825,472      6,810     338,740.9       2,772.5      1,153  413,139,275   5,360,508.9  cudaMemcpyAsync                  
      7.7    1,781,747,074  1,055,137       1,688.6       1,547.0      1,399   21,880,618      23,246.7  cudaLaunchKernel                 
      0.6      140,144,078          3  46,714,692.7      37,087.0     28,224  140,078,767  80,855,660.3  cudaMemGetInfo                   
      0.2       54,673,249          2  27,336,624.5  27,336,624.5    574,638   54,098,611  37,847,164.3  cudaMallocHost                   
      0.1       25,398,086          2  12,699,043.0  12,699,043.0    635,453   24,762,633  17,060,492.6  cudaFreeHost                     
      0.1       16,873,777          7   2,410,539.6     143,525.0      3,083   15,853,845   5,929,834.4  cudaMalloc                       
      0.0       10,527,215          8   1,315,901.9   1,214,730.0     24,559    3,369,309   1,214,198.7  cudaFree                         
      0.0        3,075,263          4     768,815.8     769,219.5    209,398    1,327,426     483,970.0  cuLibraryLoadData                
      0.0        2,260,349          1   2,260,349.0   2,260,349.0  2,260,349    2,260,349           0.0  cuMemUnmap                       
      0.0        1,974,822          2     987,411.0     987,411.0    922,431    1,052,391      91,895.6  cudaGetDeviceProperties_v2_v12000
      0.0          718,345          2     359,172.5     359,172.5     66,360      651,985     414,099.4  cuMemSetAccess                   
      0.0          582,601         68       8,567.7         148.5        143      569,861      69,082.9  cuKernelGetFunction              
      0.0          527,918        101       5,226.9       1,402.0        793      161,138      17,268.8  cudaEventRecord                  
      0.0          332,472         68       4,889.3       1,609.0      1,390       30,410       8,298.4  cuLaunchKernel                   
      0.0          136,102        838         162.4         138.0         73        4,072         154.4  cuGetProcAddress_v2              
      0.0           76,657          6      12,776.2      11,467.5      3,590       24,162       7,379.2  cudaMemsetAsync                  
      0.0           68,629          2      34,314.5      34,314.5     22,629       46,000      16,525.8  cuMemCreate                      
      0.0           39,030          1      39,030.0      39,030.0     39,030       39,030           0.0  cudaStreamDestroy                
      0.0           31,947        606          52.7          43.0         37          984          48.9  cuStreamGetCaptureInfo_v2        
      0.0           21,251          4       5,312.8       3,251.5      1,966       12,782       5,060.8  cudaDeviceSynchronize            
      0.0           17,689         18         982.7         762.5        479        4,059         808.4  cudaEventDestroy                 
      0.0           15,733          5       3,146.6       3,520.0        444        4,435       1,593.2  cuLibraryGetKernel               
      0.0           15,477          1      15,477.0      15,477.0     15,477       15,477           0.0  cuMemAddressFree                 
      0.0           13,237          1      13,237.0      13,237.0     13,237       13,237           0.0  cudaStreamCreateWithFlags        
      0.0            9,735          1       9,735.0       9,735.0      9,735        9,735           0.0  cuMemAddressReserve              
      0.0            9,523         18         529.1         290.0        229        2,234         590.5  cudaEventCreateWithFlags         
      0.0            5,787          2       2,893.5       2,893.5      2,639        3,148         359.9  cuMemMap                         
      0.0            4,399          4       1,099.8       1,249.5        626        1,274         316.0  cuInit                           
      0.0            4,155          1       4,155.0       4,155.0      4,155        4,155           0.0  cudaEventQuery                   
      0.0            3,589          1       3,589.0       3,589.0      3,589        3,589           0.0  cuMemGetAllocationGranularity    
      0.0            1,116          3         372.0         164.0        132          820         388.3  cuModuleGetLoadingMode           
      0.0              990          2         495.0         495.0        331          659         231.9  cudaGetDriverEntryPoint_v11030   
      0.0              353          2         176.5         176.5        145          208          44.5  cuMemRelease                     

Processing [dev1-gemma-4b.sqlite] with [/usr/local/cuda-12.9/nsight-systems-2025.1.3/host-linux-x64/reports/cuda_gpu_kern_sum.py]... 

 ** CUDA GPU Kernel Summary (cuda_gpu_kern_sum):

 Time (%)  Total Time (ns)  Instances  Avg (ns)   Med (ns)   Min (ns)  Max (ns)   StdDev (ns)                                                  Name                                                
 --------  ---------------  ---------  ---------  ---------  --------  ---------  -----------  ----------------------------------------------------------------------------------------------------
     86.2   16,650,623,556    253,109   65,784.4   20,736.0    10,976  5,023,823    194,450.8  void mul_mat_vec<__half, __half, (int)1, (int)256>(const T1 *, const float *, const int *, float *,…
      5.0      958,434,630     36,040   26,593.6   25,632.0    21,088  1,551,515     30,793.9  void flash_attn_ext_f16<(int)256, (int)256, (int)4, (int)2, (int)2, (int)1, (bool)0, (bool)0>(const…
      2.2      416,796,498    217,505    1,916.3    1,952.0     1,663     20,928        236.7  void k_bin_bcast<&op_mul, float, float, float>(const T2 *, const T3 *, T4 *, int, int, int, int, in…
      2.1      407,080,542    145,357    2,800.6    2,720.0     2,399     17,472        383.8  void rms_norm_f32<(int)1024>(const float *, float *, int, long, long, long, float)                  
      1.2      225,183,495     36,040    6,248.2    6,240.0     5,120    370,527      2,037.9  void flash_attn_stream_k_fixup<(int)256, (int)4, (int)2>(float *, const float2 *, int, int, int)    
      0.7      138,333,795     72,148    1,917.4    2,112.0     1,567     18,144        351.3  void rms_norm_f32<(int)32>(const float *, float *, int, long, long, long, float)                    
      0.7      131,226,534     72,148    1,818.9    1,824.0     1,504     17,440        154.4  void k_bin_bcast<&op_add, float, float, float>(const T2 *, const T3 *, T4 *, int, int, int, int, in…
      0.7      129,531,394     74,270    1,744.1    1,696.0     1,631     17,856        276.2  void cpy_f32_f16<&cpy_1_f32_f16>(const char *, char *, int, int, int, int, int, int, int, int, int,…
      0.5       93,407,271     72,148    1,294.7    1,280.0     1,184     20,127        150.0  void rope_neox<(bool)1, (bool)0, float>(const T3 *, T3 *, int, int, int, int, int, const int *, flo…
      0.4       68,189,621     36,074    1,890.3    1,856.0     1,760     18,144        364.0  void unary_gated_op_kernel<&op_gelu, float>(const T2 *, const T2 *, T2 *, long, long, long, long)   
      0.2       40,957,914     37,135    1,102.9    1,088.0       992     13,600        238.3  scale_f32(const float *, float *, float, int)                                                       
      0.1       14,449,912        235   61,489.0   22,688.0    12,448  1,269,181    110,896.0  void mul_mat_vec<__half, __half, (int)2, (int)256>(const T1 *, const float *, const int *, float *,…
      0.1       12,504,441         66  189,461.2  164,304.0   159,168  1,805,209    201,962.9  turing_h1688gemm_256x64_sliced1x2_ldg8_tn                                                           
      0.1       10,182,623        101  100,818.0   39,296.0    30,816  1,614,491    201,661.3  turing_h1688gemm_256x64_ldg8_stages_32x1_tn                                                         
      0.0        3,792,981      2,122    1,787.5    1,984.0     1,472      2,656        265.1  void k_get_rows_float<float, float>(const T1 *, const int *, T2 *, long, long, unsigned long, unsig…
      0.0        1,759,995         68   25,882.3   25,968.0    25,056     26,368        300.3  void cutlass::Kernel2<cutlass_75_wmma_tensorop_h161616gemm_32x32_128x2_tn_align8>(T1::Params)       
      0.0        1,740,533         34   51,192.1   50,591.5    49,056     67,776      3,121.1  void flash_attn_ext_f16<(int)256, (int)256, (int)16, (int)2, (int)4, (int)2, (bool)0, (bool)0>(cons…
      0.0        1,079,070        235    4,591.8    3,456.0     2,208      9,023      2,622.3  void convert_unary<__half, float>(const void *, T2 *, long, long, long, long, long, long)           
      0.0        1,016,827        235    4,326.9    3,648.0     3,008     10,304      1,904.0  void convert_unary<float, __half>(const void *, T2 *, long, long, long, long, long, long)           
      0.0          582,528         34   17,133.2   17,120.0    17,056     17,568         87.1  void flash_attn_stream_k_fixup<(int)256, (int)16, (int)2>(float *, const float2 *, int, int, int)   
      0.0          478,112        101    4,733.8    4,480.0     3,744     12,096      1,080.3  void cublasLt::splitKreduce_kernel<(int)32, (int)16, int, __half, __half, __half, __half, (bool)0, …

Processing [dev1-gemma-4b.sqlite] with [/usr/local/cuda-12.9/nsight-systems-2025.1.3/host-linux-x64/reports/cuda_gpu_mem_time_sum.py]... 

 ** CUDA GPU MemOps Summary (by Time) (cuda_gpu_mem_time_sum):

 Time (%)  Total Time (ns)  Count   Avg (ns)     Med (ns)    Min (ns)   Max (ns)    StdDev (ns)           Operation          
 --------  ---------------  -----  -----------  -----------  --------  -----------  -----------  ----------------------------
     87.6    2,527,438,955  5,749    439,631.1      1,600.0       319  413,623,340  5,864,457.8  [CUDA memcpy Host-to-Device]
     11.7      338,976,207  1,061    319,487.5    319,103.0   318,719      332,639      1,401.7  [CUDA memcpy Device-to-Host]
      0.7       18,757,906      6  3,126,317.7  2,499,002.0   320,127    7,067,246  3,163,909.3  [CUDA memset]               


```



## 性能分析结果

### CUDA API 分析

| 时间占比 | API 名称              | 调用次数  | 平均时间(ns) | 最大时间(ns) | 说明             |
| -------- | --------------------- | --------- | ------------ | ------------ | ---------------- |
| 81.2%    | cudaStreamSynchronize | 13,185    | 1,424,683    | 26,541,226   | 同步操作主导开销 |
| 10.0%    | cudaMemcpyAsync       | 6,810     | 338,741      | 413,139,275  | 异步内存拷贝     |
| 7.7%     | cudaLaunchKernel      | 1,055,137 | 1,689        | 21,880,618   | 内核启动开销低   |

**关键发现**：

1. `cudaStreamSynchronize` 占比过高（81.2%），表明存在严重的CPU-GPU同步瓶颈
2. 内存拷贝最大耗时413ms，存在大块数据传输问题
3. 内核启动效率高（平均1.69μs），说明内核调度机制良好

### GPU 内核性能分析

| 时间占比 | 内核名称                                | 调用次数 | 平均时间(μs) | 说明           |
| -------- | --------------------------------------- | -------- | ------------ | -------------- |
| 86.2%    | `void mul_mat_vec<__half, __half, ...>` | 253,109  | 65.78        | 矩阵乘法核心   |
| 5.0%     | `void flash_attn_ext_f16<...>`          | 36,040   | 26.59        | FlashAttention |
| 2.2%     | `void k_bin_bcast<&op_mul, ...>`        | 217,505  | 1.92         | 元素级操作     |

**关键发现**：

1. 矩阵乘法内核 (`mul_mat_vec`) 主导计算（86.2% GPU时间）
2. FlashAttention 优化有效（仅占5%）
3. 元素级操作高效（平均1.92μs）



