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




//gpu
__global__ void vector_add(const float* a,const float* b,float* c,int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i<n)
    {
        c[i]=a[i]+b[i];
    }
}



int main()

{
const int N =8;
const size_t bytes = N * sizeof(float);

//cpu 数据
float h_a[N]={1,2,3,4,5,6,7,8};
float h_b[N]={21,22,23,24,25,26,27,28};
float h_c[N]={};

//gpu 数据
float *d_a=nullptr;
float *d_b=nullptr;
float *d_c=nullptr;

//申请gpu显存
check_cuda(cudaMalloc((void**)&d_a,bytes),"d_a");
check_cuda(cudaMalloc((void**)&d_b,bytes),"d_b");
check_cuda(cudaMalloc((void**)&d_c,bytes),"d_c");

//cpu--gpu

check_cuda(cudaMemcpy(d_a,h_a,bytes,cudaMemcpyHostToDevice),"copy a");
check_cuda(cudaMemcpy(d_b,h_b,bytes,cudaMemcpyHostToDevice),"copy b");




//启动GPU kernel
const int threads=4;
const int blocks = (N+threads-1)/ threads;

vector_add<<<blocks,threads>>>(d_a,d_b,d_c,N);

check_cuda(cudaGetLastError(),"kernel launch");
check_cuda(cudaDeviceSynchronize(),"kernel execute");

check_cuda(cudaMemcpy(h_c,d_c,bytes,cudaMemcpyDeviceToHost),"copy c");

// 验证 GPU 结果
    bool pass = true;

    for (int i = 0; i < N; i++)
    {
        float expected = h_a[i] + h_b[i];

        printf("C[%d] = %.1f\n", i, h_c[i]);

        if (h_c[i] != expected)
        {
            pass = false;
        }
    }
// 释放 GPU 显存
    check_cuda(cudaFree(d_a), "free A");
    check_cuda(cudaFree(d_b), "free B");
    check_cuda(cudaFree(d_c), "free C");

    printf("Test: %s\n", pass ? "PASS" : "FAIL");

    return pass ? 0 : 1;
}


