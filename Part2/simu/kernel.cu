#include <stdio.h>
#include "kernel.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "util.h"

#define MASS .1f
#define SHEAR_LEN 1.414213562f
#define BEND_LEN 2.0f
#define DAMPING 0.2f
#define G 0.98f
#define K 15.0f

float* d_vertexData = 0;

glm::vec3* d_vel = 0;
glm::vec3* d_accel = 0;
glm::vec3* d_pos = 0;
glm::vec3* d_pos2 = 0;

__global__ void copyVertexDatakernel( glm::vec3* pos, glm::vec3* pos2, float4* vertexData, int massXNum, int massYNum )
{
    int2 idx;
    int offset;
    idx.x = blockIdx.x * blockDim.x + threadIdx.x;
    idx.y = blockIdx.y * blockDim.y + threadIdx.y;
    offset = idx.y * massXNum + idx.x;
    if( idx.x < massXNum && idx.y < massYNum )
    {
        pos[offset].x = vertexData[ offset ].x;
        pos[offset].y = vertexData[ offset ].y;
        pos[offset].z = vertexData[ offset ].z;
        pos2[offset] = pos[offset];
    }
}

int initCUDABuffer( cudaGraphicsResource* &vboResource, int massXNum, int massYNum )
{
    cudaErrorCheck( cudaMalloc((void**)&d_vel, massXNum * massYNum * sizeof( glm::vec3 ) ) );
    cudaErrorCheck( cudaMalloc((void**)&d_accel, massXNum * massYNum * sizeof( glm::vec3 ) ) );
    cudaErrorCheck( cudaMalloc((void**)&d_pos, massXNum * massYNum * sizeof( glm::vec3 ) ) );
    cudaErrorCheck( cudaMalloc((void**)&d_pos2, massXNum * massYNum * sizeof( glm::vec3 ) ) );

    cudaErrorCheck( cudaMemset( (void*)d_vel, 0, massXNum * massYNum * sizeof( glm::vec3 ) ) );
    cudaErrorCheck( cudaMemset( (void*)d_accel, 0, massXNum * massYNum * sizeof( glm::vec3 ) ) );

    //populate the pos buffer with data from VBO
    size_t vboSize;
    dim3 blockSize = dim3(16*16);
    dim3 gridSize = dim3( (massXNum + blockSize.x-1)/blockSize.x, (massYNum + blockSize.y-1)/blockSize.y );
    cudaErrorCheck( cudaGraphicsMapResources( 1, &vboResource, 0 ) );
    cudaErrorCheck( cudaGraphicsResourceGetMappedPointer((void**) &d_vertexData, &vboSize, vboResource ) );

    copyVertexDatakernel<<<gridSize,blockSize>>>( d_pos, d_pos2, (float4*)d_vertexData, massXNum, massYNum );

    cudaErrorCheck( cudaGraphicsUnmapResources( 1, &vboResource, 0 ) );

    return 0;
}


int cleanupCUDA()
{
    if( d_vel )
        cudaFree( d_vel );
    d_vel = 0;

    if( d_accel )
        cudaFree( d_accel );
    d_accel = 0;

    if( d_pos )
        cudaFree( d_pos );
    d_pos = 0;

    if( d_pos2 )
        cudaFree( d_pos2 );
    d_pos2 = 0;

    return 0;
}

__global__ void updateVelKernel( float4* vertexData, glm::vec3* pos, glm::vec3* accel, glm::vec3* vel,
                                 float2 restlen, int massXNum, int massYNum )
{
    int2 idx;
    int2 adj;
    int offset;
    glm::vec3 vec;
    idx.x = blockIdx.x * blockDim.x + threadIdx.x;
    idx.y = blockIdx.y * blockDim.y + threadIdx.y;
    offset = idx.y * massXNum + idx.x;
    glm::vec3 force( 0.0f,0.0f,0.0f );

    if( idx.x < massXNum && idx.y < massYNum )
    {
        //internal force - stretch
        adj.x = idx.x-1;
        if( (adj.x) >= 0 )
        {
            vec = pos[ offset - 1] - pos[ offset ];
            force += K * ( vec - restlen.x * glm::normalize( vec ) ) ;

        }
        adj.x = idx.x+1;
        if( ( adj.x ) < massXNum )
        {
            vec = pos[ idx.y * massXNum + adj.x] - pos[ offset ];
            force += K * ( vec - restlen.x * glm::normalize( vec ) );
        }
        adj.y = idx.y - 1;
        if( (adj.y) >= 0 )
        {
            vec = pos[ adj.y * massXNum + idx.x] - pos[ offset ];
            force += K * ( vec - restlen.x * glm::normalize( vec ) );
        }
        adj.y = idx.y+1;
        if( ( adj.y) < massYNum )
        {
            vec = pos[ adj.y * massXNum + idx.x] - pos[ offset ];
            force += K * ( vec - restlen.x * glm::normalize( vec ) );
        }

        //internal force - shear
        if( (adj.x = idx.x -1 ) >= 0 && (adj.y=idx.y-1) >=0 )
        {
            vec = pos[ adj.y * massXNum + adj.x] - pos[ offset ];
            force += K * ( vec - SHEAR_LEN*restlen.x * glm::normalize( vec ) );
        }

        if( (adj.x = idx.x +1 ) < massXNum && (adj.y=idx.y+1) < massYNum )
        {
            vec = pos[ adj.y * massXNum + adj.x] - pos[ offset ];
            force += K * ( vec - SHEAR_LEN*restlen.x * glm::normalize( vec ) );
        }

        if( (adj.x = idx.x +1 ) < massXNum && (adj.y=idx.y-1) >= 0)
        {
            vec = pos[ adj.y * massXNum + adj.x] - pos[ offset ];
            force += K * ( vec - SHEAR_LEN*restlen.x * glm::normalize( vec ) );
        }

        if( (adj.x = idx.x -1 ) >= 0 && (adj.y=idx.y+1) < massYNum )
        {
            vec = pos[ adj.y * massXNum + adj.x] - pos[ offset ];
            force += K * ( vec - SHEAR_LEN*restlen.x * glm::normalize( vec ) );
        }

        ////if( idx.x == 0 || idx.x == massXNum-1 || idx.y == 0 || idx.y == massYNum-1 )
        {
            //internal force - bend
            if( (adj.x = idx.x-2) >= 0 )
            {
                vec = pos[ idx.y * massXNum + adj.x] - pos[ offset ];
                force += K * ( vec - BEND_LEN*restlen.x * glm::normalize( vec ) );
            }
            if( ( adj.x = idx.x+2 ) < massXNum )
            {
                vec = pos[ idx.y * massXNum + adj.x] - pos[ offset ];
                force += K * ( vec - BEND_LEN*restlen.x * glm::normalize( vec ) );
            }
            if( (adj.y = idx.y-2) >= 0 )
            {
                vec = pos[ adj.y * massXNum + idx.x] - pos[ offset ];
                force += K * ( vec - BEND_LEN*restlen.x * glm::normalize( vec ) );
            }
            if( ( adj.y = idx.y+2 ) < massYNum )
            {
                vec = pos[ adj.y * massXNum + idx.x] - pos[ offset ];
                force += K * ( vec - BEND_LEN*restlen.x * glm::normalize( vec ) );
            }
        }
        accel[offset] = force;
        
    }
}

void updateVelWrapper( cudaGraphicsResource* &vboResource, float2 restlen, int massXNum, int massYNum )
{
    size_t vboSize;

    dim3 blockSize = dim3(16*16);
    dim3 gridSize = dim3( (massXNum + blockSize.x-1)/blockSize.x, (massYNum + blockSize.y-1)/blockSize.y );

    cudaErrorCheck( cudaGraphicsMapResources( 1, &vboResource, 0 ) );
    cudaErrorCheck( cudaGraphicsResourceGetMappedPointer((void**) &d_vertexData, &vboSize, vboResource ) );

    updateVelKernel<<<gridSize,blockSize>>>( (float4*)d_vertexData, d_pos, d_accel, d_vel, restlen, massXNum, massYNum );

    cudaErrorCheck( cudaGraphicsUnmapResources( 1, &vboResource, 0 ) );
}

__global__ void updatePosKernel( float4* vertexData, glm::vec3* pos, glm::vec3* pos2, glm::vec3* accel, glm::vec3* vel, float dt, int massXNum, int massYNum )
{
    int2 idx;
    int offset;
    glm::vec3 newPos;
    glm::vec3 totalForce;
    idx.x = blockIdx.x * blockDim.x + threadIdx.x;
    idx.y = blockIdx.y * blockDim.y + threadIdx.y;
    offset = idx.y * massXNum + idx.x;

    if( idx.x < massXNum && idx.y < massYNum )
    {
        if( ( idx.x == 0 || idx.x == massXNum-1 ) && idx.y == 0 )
            return;
        //add external forces
        //Gravity
        totalForce = accel[offset] + MASS * glm::vec3( 0, -G, 0 );
        totalForce -= DAMPING*(pos[offset] - pos2[offset])/dt;

        newPos = 2.0f*pos[offset] - pos2[offset] + (dt*dt*totalForce/MASS);

        pos2[offset] = pos[offset];
        pos[offset] = newPos;
        
        vertexData[ offset ].x = newPos.x;
        vertexData[ offset ].y = newPos.y;
        vertexData[ offset ].z = newPos.z;
    }
}

void updatePosWrapper( cudaGraphicsResource* &vboResource, float dt, int massXNum, int massYNum )
{
    size_t vboSize;

    dim3 blockSize = dim3(16*16);
    dim3 gridSize = dim3( (massXNum + blockSize.x-1)/blockSize.x, (massYNum + blockSize.y-1)/blockSize.y );

    cudaErrorCheck( cudaGraphicsMapResources( 1, &vboResource, 0 ) );
    cudaErrorCheck( cudaGraphicsResourceGetMappedPointer((void**) &d_vertexData, &vboSize, vboResource ) );

    updatePosKernel<<<gridSize,blockSize>>>( (float4*)d_vertexData,d_pos, d_pos2, d_accel, d_vel, dt,  massXNum, massYNum );

    cudaErrorCheck( cudaGraphicsUnmapResources( 1, &vboResource, 0 ) );
}