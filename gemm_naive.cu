#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>



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


void gemm_cpu(const float* A,const float* B,float* C,int M,int K,int N)
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
    const int M = 2;
    const int K = 3;
    const int N = 2;

    float h_A[M*K] = {
        1, 2, 3,
        4, 5, 6
    };

    float h_B[K*N] = {
        7, 8,   
        9, 10,
        11, 12
    };

    float h_C[M*N] = {};

    // CPU 矩阵乘法
    gemm_cpu(h_A, h_B, h_C, M, K, N);

    // 输出结果
    for (int row = 0; row < M; ++row)
    {
        for (int col = 0; col < N; ++col)
        {
            printf("%.1f ", h_C[row*N+col]);
        }

        printf("\n");
    }

    //GPU部分

    float h_c[M*N]={};

    float* d_a=nullptr;
    float* d_b=nullptr;
    float* d_c=nullptr;

const size_t bytes_a = M * K * sizeof(float);
const size_t bytes_b = K * N * sizeof(float);
const size_t bytes_c = M * N * sizeof(float);

check_cuda(cudaMalloc((void**)&d_a, bytes_a), "malloc A");
check_cuda(cudaMalloc((void**)&d_b, bytes_b), "malloc B");
check_cuda(cudaMalloc((void**)&d_c, bytes_c), "malloc C");

check_cuda(cudaMemcpy(d_a,h_A,bytes_a,cudaMemcpyHostToDevice),"copy a");
check_cuda(cudaMemcpy(d_b,h_B,bytes_b,cudaMemcpyHostToDevice),"copy b");

//启动GPU kernel
const int threads=4;
const int blocks = (M*N+threads-1)/ threads;


gemm_naive<<<blocks,threads>>>(d_a,d_b,d_c,M,K,N);
check_cuda(cudaGetLastError(),"kernel launch");
check_cuda(cudaDeviceSynchronize(),"kernel execute");

check_cuda(cudaMemcpy(h_c,d_c,bytes_c,cudaMemcpyDeviceToHost),"copy c");

bool pass = true;

for (int i = 0; i < M * N; ++i)
{
    printf("CPU[%d] = %.1f, GPU[%d] = %.1f\n",
           i, h_C[i], i, h_c[i]);

    if (h_C[i] != h_c[i])
    {
        pass = false;
    }
}

printf("Test: %s\n", pass ? "PASS" : "FAIL");

check_cuda(cudaFree(d_a), "free A");
check_cuda(cudaFree(d_b), "free B");
check_cuda(cudaFree(d_c), "free C");

    return pass ? 0 : 1;
}