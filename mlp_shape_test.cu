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


    const int BATCH=8;    //2行  1行是一个样本    输入行
    const int IN=16;   //3列 一个样本3个特征     输入列
    const int HIDDEN=32;   //隐藏层 4个特组      w1列
    const int OUT=8;  //输出样本 2个值

    std::vector<float> h_X(BATCH * IN);
    std::vector<float> h_W1(IN * HIDDEN);
    std::vector<float> h_W2(HIDDEN * OUT);
    std::vector<float> h_H(BATCH * HIDDEN,0.0f);
    std::vector<float> h_Y(BATCH * OUT,0.0f);
    std::vector<float> h_y(BATCH * OUT,0.0f);
   
// 初始化输入 X
for (int i = 0; i < BATCH * IN; ++i)
{
    h_X[i] = (i % 7 - 3) * 0.25f;
}

// 初始化 W1
for (int i = 0; i < IN * HIDDEN; ++i)
{
    h_W1[i] = (i % 9 - 4) * 0.125f;
}

// 初始化 W2
for (int i = 0; i < HIDDEN * OUT; ++i)
{
    h_W2[i] = (i % 11 - 5) * 0.0625f;
}


// CPU 中间结果


// CPU 最终结果




    // CPU 矩阵乘法
    gemm_cpu(h_X.data(), h_W1.data(), h_H.data(), BATCH, IN, HIDDEN);

    relu_cpu(h_H.data(), BATCH * HIDDEN);

    gemm_cpu(h_H.data(), h_W2.data(), h_Y.data(), BATCH, HIDDEN, OUT);


printf("CPU MLP first values:\n");

const int output_count = BATCH * OUT;
const int print_count = output_count < 8 ? output_count : 8;

for (int i = 0; i < print_count; ++i)
{
    printf("%.6f ", h_Y[i]);
}
printf("\n");
   
//GPU
   
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

check_cuda(cudaMemcpy(d_X,h_X.data(),bytes_d_X,cudaMemcpyHostToDevice),"copy h_X");
check_cuda(cudaMemcpy(d_W1,h_W1.data(),bytes_d_W1,cudaMemcpyHostToDevice),"copy h_W1");
check_cuda(cudaMemcpy(d_W2,h_W2.data(),bytes_d_W2,cudaMemcpyHostToDevice),"copy h_W2");



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

check_cuda(cudaMemcpy(h_y.data(),d_Y,bytes_d_Y,cudaMemcpyDeviceToHost),"copy c");


float total_ms=0.0f;
float avg_ms=0.0f;
check_cuda(cudaEventElapsedTime(&total_ms, start, stop),"total_ms time");
avg_ms=total_ms / repeat;
printf("Total GPU Time: %.6f ms\n", total_ms);
printf("Average GPU MLP Time: %.6f ms\n", avg_ms);

bool pass = true;

const float atol = 1e-5f;
const float rtol = 1e-4f;

float max_diff = 0.0f;

for (int i = 0; i < BATCH * OUT; ++i)
{
    float cpu = h_Y[i];
    float gpu = h_y[i];

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

printf("Checked Elements: %d\n", BATCH * OUT);
printf("Max Absolute Difference: %.8f\n", max_diff);
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
