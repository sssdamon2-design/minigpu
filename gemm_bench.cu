#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <vector>
#include <cmath>

//检查cuda API返回值
void check_cuda(cudaError_t err, const char* step)
{
    if (err != cudaSuccess)
    {
        printf("%s failed: %s\n",
               step, cudaGetErrorString(err));
        std::exit(1);
    }
}



__global__ void gemm_naive(const float* A,const float* B,float* C,int M,int K,int N)
{
    int idx= blockIdx.x * blockDim.x + threadIdx.x;
    
    if(idx>=M*N)
    return;

    int row=idx / N;
    int col=idx % N;

    float sum=0.0f;

    for (int k=0;k<K;k++)
    {
        sum+=A[row*K+k]*B[k*N+col];
    }
    C[row*N+col]=sum;

}


void gemm_cpu(const float* A,const float* B,float* C,int M,int K,int N)// 矩阵ABC   A[M,K] B[K,N] C[M,N]
{
    for (int row = 0; row < M; ++row)
    {
        for (int col = 0; col < N; ++col)
        {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k)
            {
                sum += A[row*K+k]*B[k*N+col];
            }
            C[row*N+col]=sum;
            
        }
    }
}


int main()
{
   


    const int M=130;    
    const int K=65;  
    const int N=70;   

    std::vector<float> h_A(M * K);
    std::vector<float> h_B(K * N);
    std::vector<float> h_C_cpu(M * N);
    std::vector<float> h_C_gpu(M *N,0.0f);


    for (int i = 0; i < M * K; ++i)
{
    h_A[i] = (i % 7 - 3) * 0.25f;
}

for (int i = 0; i < K * N; ++i)
{
    h_B[i] = (i % 9 - 4) * 0.125f;
}
  
//cpu执行
gemm_cpu(h_A.data(),h_B.data(),h_C_cpu.data(),M, K, N);



//GPU执行
float* d_A=nullptr;
float* d_B=nullptr;
float* d_C=nullptr;

const size_t bytes_d_A=M * K* sizeof(float);
const size_t bytes_d_B=K * N* sizeof(float);
const size_t bytes_d_C=M * N* sizeof(float);


check_cuda(cudaMalloc((void**)&d_A,bytes_d_A),"malloc d_A");
check_cuda(cudaMalloc((void**)&d_B,bytes_d_B),"malloc d_B");
check_cuda(cudaMalloc((void**)&d_C,bytes_d_C),"malloc d_C");


check_cuda(cudaMemcpy(d_A,h_A.data(),bytes_d_A,cudaMemcpyHostToDevice),"Memcpy d_A");
check_cuda(cudaMemcpy(d_B,h_B.data(),bytes_d_B,cudaMemcpyHostToDevice),"Memcpy d_B");
  


const int threads=128;
const int blocks=(M * N+threads-1)/threads;

cudaEvent_t start, stop;

check_cuda(cudaEventCreate(&start), "create start");
check_cuda(cudaEventCreate(&stop), "create stop");


const int warmup = 20;
const int repeat = 1000;

for (int i = 0; i < warmup; ++i)
{
    gemm_naive<<<blocks, threads>>>(d_A, d_B, d_C, M, K, N);
}

check_cuda(cudaGetLastError(), "warmup launch");
check_cuda(cudaDeviceSynchronize(), "warmup sync");


check_cuda(cudaEventRecord(start),"record start");

for (int i = 0; i < repeat; ++i)
{
    gemm_naive<<<blocks, threads>>>(d_A, d_B, d_C, M, K, N);
}
check_cuda(cudaGetLastError(), "benchmark kernel launch");
check_cuda(cudaEventRecord(stop),"record stop"); //在这里记录stop时间，但是cpu执行到这里的时候gpu不一定执行到这里了
check_cuda(cudaEventSynchronize(stop),"sync stop");  //设置同步，cpu要等待GPU

float total_ms = 0.0f;
float avg_ms = 0.0f;

check_cuda(cudaEventElapsedTime(&total_ms, start, stop),"elapsed time");
avg_ms=total_ms/repeat;

double flops = 2.0 * M * N * K;
double gflops = flops / (avg_ms * 1e6);

printf("GEMM Shape: M=%d, K=%d, N=%d\n", M, K, N);
printf("Threads Per Block: %d\n", threads);
printf("Warmup: %d, Repeat: %d\n", warmup, repeat);
printf("Total GPU Time: %.6f ms\n", total_ms);
printf("Average GEMM Time: %.6f ms\n", avg_ms);
printf("Performance: %.3f GFLOPS\n", gflops);

check_cuda(cudaMemcpy(h_C_gpu.data(),d_C,bytes_d_C,cudaMemcpyDeviceToHost),"Memcpy h_C_gpu");

bool pass = true;

const float atol = 1e-5f;
const float rtol = 1e-4f;

float max_diff = 0.0f;

for (int i = 0; i < M * N; ++i)
{
    float cpu = h_C_cpu[i];
    float gpu = h_C_gpu[i];

    float diff = std::fabs(cpu - gpu);
    float tolerance = atol + rtol * std::fabs(cpu);

    if (diff > max_diff)
    {
        max_diff = diff;
    }

    // 只打印前 8 个元素，但检查所有元素
    if (i < 8)
    {
        printf("CPU[%d] = %.6f, GPU[%d] = %.6f, Diff = %.8f\n",
               i, cpu, i, gpu, diff);
    }

    if (!std::isfinite(cpu) || !std::isfinite(gpu) ||
        diff > tolerance)
    {
        pass = false;
    }
}

printf("Checked Elements: %d\n", M * N);
printf("Max Absolute Difference: %.8f\n", max_diff);
printf("Test: %s\n", pass ? "PASS" : "FAIL");

    check_cuda(cudaFree(d_A),  "free d_A");
    check_cuda(cudaFree(d_B), "free d_B");
    check_cuda(cudaFree(d_C),  "free d_C");

check_cuda(cudaEventDestroy(start), "destroy start");
check_cuda(cudaEventDestroy(stop), "destroy stop");
    return pass ? 0 : 1;
}



