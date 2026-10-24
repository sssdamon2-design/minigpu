#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <vector>
#include <cmath>
constexpr int TILE = 32;
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

__global__ void gemm_tiled(const float* A,const float* B,float* C,int M,int K,int N)//这个函数是造shared缓存的A和B
{
    int tx = threadIdx.x;              //当前线程的块内横坐标， 也是目前AS BS矩阵中  目前线程计算的位置坐标
    int ty = threadIdx.y;              //当前线程的块内纵坐标   也是目前AS BS矩阵中  目前线程计算的位置坐标

    int row = blockIdx.y * blockDim.y + ty;         //当前线程的C内纵坐标
    
    int col = blockIdx.x * blockDim.x + tx;         //当前线程的C内横坐标

   __shared__ float As[TILE][TILE];             //需要把A B 赋值过去的 shared存储的AS BS 
   __shared__ float Bs[TILE][TILE];             //后续从AS BS 提取需要计算的值
    
    float sum=0.0f;

    for(int t=0;  t<(K+TILE-1)/TILE ;  ++t )   //  t是在AB上取tile的滑动，每次滑动TILE，A横轴滑动B竖轴滑动
    {
        //一个循环滑动一次  一个block计算一个tile  分清计算shared存贮         shared值的存贮需要K个乘积值相加   ，所以aCOL小于K
        int aCol=t* TILE + tx;          //用滑动次数t和TIle + 当前线程的块内横坐标 计算As的列   As的行就是block线程对应的行
        int bRow=t* TILE + ty;          //用滑动次数t和TIle + 当前线程的块内纵坐标  计算Bs的行   Bs的列就是block中线程对应的列

        
        if(aCol < K && row< M)
        {
        As[ty][tx]=A[row*K+aCol];
        }
        else
        As[ty][tx]=0.0f;

        if(bRow<K && col <N)
        {
        Bs[ty][tx]=B[bRow*N+col];
        }
        else
        Bs[ty][tx]=0.0f;

        __syncthreads();

        for(int k=0;k<TILE;k++)
        {
            sum+=As[ty][k] * Bs[k][tx] ;
        }

        __syncthreads();

    }
         // 结束 t 循环

    // 所有 K 方向的 Tile 计算完成
    // 每个有效线程把自己的 sum 写回 C
    if (row < M && col < N)
    {
        C[row * N + col] = sum;
    }





}


__global__ void gemm_register_2(const float* A,const float* B,float* C,int M,int K,int N)//这个函数是造shared缓存的A和B
{
    int tx = threadIdx.x;              //当前线程的块内横坐标， 也是目前AS BS矩阵中  目前线程计算的位置坐标
    int ty = threadIdx.y;              //当前线程的块内纵坐标   也是目前AS BS矩阵中  目前线程计算的位置坐标

    int row = blockIdx.y * TILE + ty;         //当前线程的C内纵坐标
    
    int col0 = blockIdx.x * (TILE * 2) + tx;         //当前线程的C内横坐标
    int col1 = col0 + TILE ;
   __shared__ float As[TILE][TILE];             //需要把A B 赋值过去的 shared存储的AS BS 
   __shared__ float Bs[TILE][TILE*2];             //后续从AS BS 提取需要计算的值
    
    float sum0=0.0f;
    float sum1=0.0f;
    for(int t=0;  t<(K+TILE-1)/TILE ;  ++t )   //  t是在AB上取tile的滑动，每次滑动TILE，A横轴滑动B竖轴滑动
    {
        //一个循环滑动一次  一个block计算一个tile  分清计算shared存贮         shared值的存贮需要K个乘积值相加   ，所以aCOL小于K
        int aCol=t* TILE + tx;          //用滑动次数t和TIle + 当前线程的块内横坐标 计算As的列   As的行就是block线程对应的行
        int bRow=t* TILE + ty;          //用滑动次数t和TIle + 当前线程的块内纵坐标  计算Bs的行   Bs的列就是block中线程对应的列

        
        if(aCol < K && row< M)
        {
        As[ty][tx]=A[row*K+aCol];
        }
        else
        As[ty][tx]=0.0f;

        if(bRow<K && col0 <N)
        {
        Bs[ty][tx]=B[bRow*N+col0];
        }
        else
        Bs[ty][tx]=0.0f;

        if(bRow<K && col1 <N)
        {
            Bs[ty][tx+TILE]=B[bRow*N+col1];

        }
        else
        Bs[ty][tx+TILE]=0.0f;
        __syncthreads();

        for(int k=0;k<TILE;k++)
        {
            sum0+=As[ty][k] * Bs[k][tx] ;
            sum1+=As[ty][k] * Bs[k][tx+TILE] ;
        }

        __syncthreads();

    }
         // 结束 t 循环

    // 所有 K 方向的 Tile 计算完成
    // 每个有效线程把自己的 sum 写回 C
    if (row < M && col0 < N)
    {
        C[row * N + col0] = sum0;
    }

    if (row < M && col1 < N)
    {
        C[row * N + col1] = sum1;
    }



}



int main(int argc, char* argv[])
{
    int M = 512;
    int K = 512;
    int N = 512;

    if (argc == 4)
    {
        M = std::atoi(argv[1]);
        K = std::atoi(argv[2]);
        N = std::atoi(argv[3]);

        if (M <= 0 || K <= 0 || N <= 0)
        {
            printf("Error: M, K, N must be positive\n");
            return 1;
        }
    }
    else if (argc != 1)
    {
        printf("Usage: %s [M K N]\n", argv[0]);
        return 1;
    }   

    std::vector<float> h_A(M * K);
    std::vector<float> h_B(K * N);
    std::vector<float> h_C_cpu(M * N);
    std::vector<float> h_C_gpu(M *N,0.0f);
    std::vector<float> h_C_tiled(M * N, 0.0f);

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

// Tiled GEMM 二维线程配置
dim3 reg_block(TILE, TILE);

dim3 reg_grid(
    (N + 2*TILE - 1) / (2*TILE),
    (M + TILE - 1) / TILE
); //个数

int tiled_active = 0;
int reg_active = 0;

// 查询 V2.2 每个 SM 最多能同时驻留多少个 Block
check_cuda(
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &tiled_active,
        gemm_tiled,
        TILE * TILE,
        0
    ),
    "tiled occupancy"
);

// 查询 V2.4
check_cuda(
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &reg_active,
        gemm_register_2,
        TILE * TILE,
        0
    ),
    "register occupancy"
);

printf("Tiled Active Blocks/SM: %d\n", tiled_active);
printf("Register2 Active Blocks/SM: %d\n", reg_active);

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
    gemm_naive<<<blocks, threads>>>(
        d_A, d_B, d_C, M, K, N
    );
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
printf("Register2 Threads Per Block: %d\n",
       reg_block.x * reg_block.y);
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


// ====================================
// V2.2 Tiled GEMM 正确性测试
// ====================================

// 测试前清零 d_C，避免保留上一次 Naive 的结果
check_cuda(cudaMemset(d_C, 0, bytes_d_C),"clear d_C");

// 启动 Tiled GEMM
gemm_register_2<<<reg_grid, reg_block>>>(d_A, d_B, d_C, M, K, N);

// 检查 Kernel 启动错误
check_cuda(cudaGetLastError(),"tiled kernel launch");

// 等待 Kernel 执行完成
check_cuda(cudaDeviceSynchronize(),"tiled kernel execute");

// GPU -> CPU
check_cuda(cudaMemcpy(h_C_tiled.data(),d_C,bytes_d_C,cudaMemcpyDeviceToHost),"copy tiled C D2H");

// 验证 Tiled GEMM 正确性
bool tiled_pass = true;

float tiled_max_diff = 0.0f;

for (int i = 0; i < M * N; ++i)
{
    float cpu = h_C_cpu[i];
    float gpu = h_C_tiled[i];

    float diff = std::fabs(cpu - gpu);

    float tolerance = atol + rtol * std::fabs(cpu);

    if (diff > tiled_max_diff)
    {
        tiled_max_diff = diff;
    }

    if (i < 8)
    {
        printf(
            "Tiled CPU[%d] = %.6f, GPU[%d] = %.6f, Diff = %.8f\n",
            i, cpu, i, gpu, diff
        );
    }

    if (!std::isfinite(cpu) ||
        !std::isfinite(gpu) ||
        diff > tolerance)
    {
        tiled_pass = false;
    }
}

printf("\n===== Tiled GEMM Result =====\n");
printf("Checked Elements: %d\n", M * N);
printf("Max Absolute Difference: %.8f\n", tiled_max_diff);
printf("Register2 Test: %s\n",
       tiled_pass ? "PASS" : "FAIL");



// ============================================
// V2.2 Tiled GEMM Benchmark
// ============================================

if (tiled_pass)
{
    // 1. 预热
    for (int i = 0; i < warmup; ++i)
    {
        gemm_register_2<<<reg_grid, reg_block>>>(d_A, d_B, d_C, M, K, N);
    }

    check_cuda(cudaGetLastError(), "tiled warmup launch");
    check_cuda(cudaDeviceSynchronize(), "tiled warmup sync");

    // 2. 开始计时
    check_cuda(cudaEventRecord(start), "tiled record start");

    // 3. 重复执行 Tiled GEMM
    for (int i = 0; i < repeat; ++i)
    {
        gemm_register_2<<<reg_grid, reg_block>>>(
            d_A, d_B, d_C, M, K, N
        );
    }

    check_cuda(cudaGetLastError(), "tiled benchmark launch");

    // 4. 停止计时并同步
    check_cuda(cudaEventRecord(stop), "tiled record stop");
    check_cuda(cudaEventSynchronize(stop), "tiled sync stop");

    // 5. 计算延迟
    float tiled_total_ms = 0.0f;

    check_cuda(
        cudaEventElapsedTime(&tiled_total_ms, start, stop),
        "tiled elapsed time"
    );

    float tiled_avg_ms = tiled_total_ms / repeat;

    // 6. 计算 GFLOPS
    double tiled_gflops = flops / (tiled_avg_ms * 1e6);

    // 7. 加速比：Naive 时间 / Tiled 时间
    double speedup = avg_ms / tiled_avg_ms;

    printf("\n===== GEMM Performance Comparison =====\n");
    printf("Shape: M=%d K=%d N=%d\n", M, K, N);

    printf("Naive Latency: %.6f ms\n", avg_ms);
    printf("Naive GFLOPS: %.3f\n", gflops);

    printf("Register2 Latency: %.6f ms\n", tiled_avg_ms);
    printf("Register2 GFLOPS: %.3f\n", tiled_gflops);

    printf("Speedup: %.3fx\n", speedup);
}












    check_cuda(cudaFree(d_A),  "free d_A");
    check_cuda(cudaFree(d_B), "free d_B");
    check_cuda(cudaFree(d_C),  "free d_C");

check_cuda(cudaEventDestroy(start), "destroy start");
check_cuda(cudaEventDestroy(stop), "destroy stop");
   return (pass && tiled_pass) ? 0 : 1;
}



