__global__ void gemm_tiled(const float* A,const float* B,float* C,int M,int K,int N)//这个函数是造shared缓存的A和B
{
    int tx = threadIdx.x;              //当前线程的块内横坐标， 也是目前AS BS矩阵中  目前线程计算的位置坐标
    int ty = threadIdy.y;              //当前线程的块内纵坐标   也是目前AS BS矩阵中  目前线程计算的位置坐标

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

        float v_a,v_b;


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



        if(row < M && col < N)
    {
        C[row*N+col]=sum;
    }


    }







int main()
{
    constexpr int TILE = 16;
    dim3 block(TILE,TILE);

    dim3 grid((N+TILE-1)/TILE,(M+TILE-1)/TILE);

}