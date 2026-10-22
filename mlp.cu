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


__global__ void relu_inplace(float* data,int count)
{
int idx= blockIdx.x * blockDim.x + threadIdx.x;
if(idx>=count)
    return;


        if(data[idx]<0.0f)
        {
        data[idx]=0;
        }
    
}

void relu_cpu(float* data, int count)
{
    for(int i=0;i<count;i++)
    {
        if(data[i]<0.0f)
{
    data[i]=0;
}

    }

  
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
    cudaEvent_t start,stop;
    check_cuda(cudaEventCreate(&start),"creat start event");
    check_cuda(cudaEventCreate(&stop), "stop start event");


    const int BATCH=2;    //2行  1行是一个样本
    const int IN=3;   //3列 一个样本3个特征
    const int HIDDEN=4;   //隐藏层 4个特组
    const int OUT=2;  //输出样本 2个值

   // 输入矩阵 X：[2,3]
float h_X[BATCH * IN] = {
    1, 2, 3,
    4, 5, 6
};

// 第一层权重 W1：[3,4]
float h_W1[IN * HIDDEN] = {
     1, -1,  0,  2,
     0,  1, -2,  0,
    -1,  0,  1, -1
};

// 第二层权重 W2：[4,2]
float h_W2[HIDDEN * OUT] = {
     1,  0,
     0,  2,
    -1,  1,
     3, -1
};

// CPU 中间结果
float h_H[BATCH * HIDDEN] = {};

// CPU 最终结果
float h_Y[BATCH * OUT] = {};



    // CPU 矩阵乘法
    gemm_cpu(h_X, h_W1, h_H, BATCH, IN, HIDDEN);

    relu_cpu(h_H, BATCH * HIDDEN);

    gemm_cpu(h_H, h_W2, h_Y, BATCH, HIDDEN, OUT);


printf("CPU MLP Output:\n");

for (int row = 0; row < BATCH; ++row)
{
    for (int col = 0; col < OUT; ++col)
    {
        printf("%.1f ", h_Y[row * OUT + col]);
    }
    printf("\n");
}
   
//GPU
    float  h_y[BATCH*OUT]={};
    float* d_X=nullptr;
    float* d_W1=nullptr;
    float* d_H=nullptr;
    float* d_W2=nullptr;
    float* d_Y=nullptr;

const size_t bytes_d_X = BATCH * IN * sizeof(float);
const size_t bytes_d_W1 = IN * HIDDEN * sizeof(float);
const size_t bytes_d_H = BATCH * HIDDEN * sizeof(float);
const size_t bytes_d_W2 = HIDDEN * OUT  * sizeof(float);
const size_t bytes_d_Y = BATCH * OUT * sizeof(float);

check_cuda(cudaMalloc((void**)&d_X, bytes_d_X),   "malloc d_X");
check_cuda(cudaMalloc((void**)&d_W1, bytes_d_W1), "malloc d_W1");
check_cuda(cudaMalloc((void**)&d_W2, bytes_d_W2), "malloc d_W2");
check_cuda(cudaMalloc((void**)&d_H, bytes_d_H), "malloc d_H");
check_cuda(cudaMalloc((void**)&d_Y, bytes_d_Y), "malloc d_Y");

check_cuda(cudaMemcpy(d_X,h_X,bytes_d_X,cudaMemcpyHostToDevice),"copy h_X");
check_cuda(cudaMemcpy(d_W1,h_W1,bytes_d_W1,cudaMemcpyHostToDevice),"copy h_W1");
check_cuda(cudaMemcpy(d_W2,h_W2,bytes_d_W2,cudaMemcpyHostToDevice),"copy h_W2");



//启动GPU
const int threads=4;
const int blocks=(BATCH*HIDDEN + threads-1)/ threads;

const int blocks2=(BATCH*OUT + threads-1)/ threads;
//预热

const int warmup=20;
const int repeat=1000;

for(int i=0;i<warmup;i++)
{
    gemm_naive<<<blocks,threads>>>(d_X,d_W1,d_H,BATCH ,IN,HIDDEN);
    relu_inplace<<<blocks,threads>>>(d_H,BATCH*HIDDEN);
    gemm_naive<<<blocks2,threads>>>(d_H,d_W2,d_Y, BATCH  ,HIDDEN,OUT);
}
check_cuda(cudaGetLastError(),"gemm_naive_warmup kernel launch");  //读取GPU错误码并清空
check_cuda(cudaDeviceSynchronize(),"GEMM _warmup launch kernel execute");

//GPU开始记录时间

check_cuda(cudaEventRecord(start),"record start");



for(int i_2=0;i_2<repeat;i_2++)
{

gemm_naive<<<blocks,threads>>>(d_X,d_W1,d_H,BATCH ,IN,HIDDEN);
relu_inplace<<<blocks,threads>>>(d_H,BATCH*HIDDEN);
gemm_naive<<<blocks2,threads>>>(d_H,d_W2,d_Y, BATCH  ,HIDDEN,OUT);

}
check_cuda(cudaEventRecord(stop),"record stop"); //在这里记录stop时间，但是cpu执行到这里的时候gpu不一定执行到这里了
check_cuda(cudaEventSynchronize(stop),"sync stop");  //设置同步，cpu要等待GPU

check_cuda(cudaMemcpy(h_y,d_Y,bytes_d_Y,cudaMemcpyDeviceToHost),"copy c");


float total_ms=0.0f;
float avg_ms=0.0f;
check_cuda(cudaEventElapsedTime(&total_ms, start, stop),"total_ms time");
avg_ms=total_ms / repeat;
printf("Total GPU Time: %.6f ms\n", total_ms);
printf("Average GPU MLP Time: %.6f ms\n", avg_ms);

bool pass = true;

for (int i = 0; i < BATCH*OUT; ++i)
{
    printf("CPU[%d] = %.1f, GPU[%d] = %.1f\n",
           i, h_Y[i], i, h_y[i]);

    if (h_Y[i] != h_y[i])
    {
        pass = false;
    }
}
printf("Test: %s\n", pass ? "PASS" : "FAIL");





    check_cuda(cudaFree(d_X),  "free d_X");
    check_cuda(cudaFree(d_W1), "free d_W1");
    check_cuda(cudaFree(d_H),  "free d_H");
    check_cuda(cudaFree(d_W2), "free d_W2");
    check_cuda(cudaFree(d_Y),  "free d_Y");





check_cuda(cudaEventDestroy(start), "destroy start");
check_cuda(cudaEventDestroy(stop), "destroy stop");


return pass ? 0 : 1;

    
}