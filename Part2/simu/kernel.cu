#include <stdio.h>
#include "kernel.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "util.h"

#define MASS 0.0001f
#define REST_LEN 1.0f
#define SHEAR_REST_LEN 1.414213562f
#define BEND_REST_LEN 2.0f
#define G 9.8f
#define K 0.5f

float* d_vertexData = 0;

glm::vec3* d_vel = 0;
glm::vec3* d_accel = 0;
glm::vec3* d_pos = 0;
glm::vec3* d_pos2 = 0;

__device__ glm::vec3 gravity( 0.0f, -G, 0.0f );

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

    cudaErrorCheck( cudaMemset( (void*)d_vel, 0, massXNum * massYNum * sizeof( glm::vec3 ) ) );
    cudaErrorCheck( cudaMemset( (void*)d_accel, 0, massXNum * massYNum * sizeof( glm::vec3 ) ) );
    cudaErrorCheck( cudaMemset( (void*)d_pos, 0, massXNum * massYNum * sizeof( glm::vec3 ) ) );

    //populate the pos buffer with data from VBO
    size_t vboSize;
    dim3 blockSize = dim3(16*16);
    dim3 gridSize = dim3( (massXNum + blockSize.x-1)/blockSize.x, (massYNum + blockSize.y-1)/blockSize.y );
    cudaErrorCheck( cudaGraphicsMapResources( 1, &vboResource, 0 ) );
    cudaErrorCheck( cudaGraphicsResourceGetMappedPointer((void**) &d_vertexData, &vboSize, vboResource ) );

    copyVertexDatakernel<<<gridSize,blockSize>>>( d_pos, d_pos, (float4*)d_vertexData, massXNum, massYNum );

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

    return 0;
}

__global__ void updateVelKernel( float4* vertexData, glm::vec3* pos, glm::vec3* vel, glm::vec3* accel, int massXNum, int massYNum )
{
    int2 idx;
    int2 adj;
    int offset;
    glm::vec3 vec;
    idx.x = blockIdx.x * blockDim.x + threadIdx.x;
    idx.y = blockIdx.y * blockDim.y + threadIdx.y;
    offset = idx.y * massXNum + idx.x;
    glm::vec3 force( 0.0f,0.0f,0.0f );

    //internal force - stretch
    if( (adj.x = idx.x-1) >= 0 )
    {
        vec = pos[ offset - 1] - pos[ offset ];
        force += -K * ( vec - REST_LEN * glm::normalize( vec ) );
    }
    if( ( adj.x = idx.x+1 ) < massXNum )
    {
        vec = pos[ idx.y * massXNum + adj.x] - pos[ offset ];
        force += -K * ( vec - REST_LEN * glm::normalize( vec ) );
    }
    if( (adj.y = idx.y-1) >= 0 )
    {
        vec = pos[ adj.y * massXNum + idx.x] - pos[ offset ];
        force += -K * ( vec - REST_LEN * glm::normalize( vec ) );
    }
    if( ( adj.y = idx.y+1 ) < massYNum )
    {
        vec = pos[ adj.y * massXNum + idx.x] - pos[ offset ];
        force += -K * ( vec - REST_LEN * glm::normalize( vec ) );
    }

    //internal force - shear
    if( (adj.x = idx.x -1 ) >= 0 && (adj.y=idx.y-1) >=0 )
    {
        vec = pos[ adj.y * massXNum + adj.x] - pos[ offset ];
        force += -K * ( vec - SHEAR_REST_LEN * glm::normalize( vec ) );
    }

    if( (adj.x = idx.x +1 ) < massXNum && (adj.y=idx.y+1) < massYNum )
    {
        vec = pos[ adj.y * massXNum + adj.x] - pos[ offset ];
        force += -K * ( vec - SHEAR_REST_LEN * glm::normalize( vec ) );
    }

    if( (adj.x = idx.x +1 ) < massXNum && (adj.y=idx.y-1) >= 0)
    {
        vec = pos[ adj.y * massXNum + adj.x] - pos[ offset ];
        force += -K * ( vec - SHEAR_REST_LEN * glm::normalize( vec ) );
    }

    if( (adj.x = idx.x -1 ) >= 0 && (adj.y=idx.y+1) < massYNum )
    {
        vec = pos[ adj.y * massXNum + adj.x] - pos[ offset ];
        force += -K * ( vec - SHEAR_REST_LEN * glm::normalize( vec ) );
    }

    //internal force - bend
    if( (adj.x = idx.x-2) >= 0 )
    {
        vec = pos[ idx.y * massXNum + adj.x] - pos[ offset ];
        force += -K * ( vec - BEND_REST_LEN * glm::normalize( vec ) );
    }
    if( ( adj.x = idx.x+2 ) < massXNum )
    {
        vec = pos[ idx.y * massXNum + adj.x] - pos[ offset ];
        force += -K * ( vec - BEND_REST_LEN * glm::normalize( vec ) );
    }
    if( (adj.y = idx.y-2) >= 0 )
    {
        vec = pos[ adj.y * massXNum + idx.x] - pos[ offset ];
        force += -K * ( vec - BEND_REST_LEN * glm::normalize( vec ) );
    }
    if( ( adj.y = idx.y+2 ) < massYNum )
    {
        vec = pos[ adj.y * massXNum + idx.x] - pos[ offset ];
        force += -K * ( vec - BEND_REST_LEN * glm::normalize( vec ) );
    }

    force += MASS * gravity;

    accel[offset] = force / MASS;

}

void upateVelWrapper( cudaGraphicsResource* &vboResource, int massXNum, int massYNum )
{
    size_t vboSize;

    dim3 blockSize = dim3(16*16);
    dim3 gridSize = dim3( (massXNum + blockSize.x-1)/blockSize.x, (massYNum + blockSize.y-1)/blockSize.y );

    cudaErrorCheck( cudaGraphicsMapResources( 1, &vboResource, 0 ) );
    cudaErrorCheck( cudaGraphicsResourceGetMappedPointer((void**) &d_vertexData, &vboSize, vboResource ) );

    updateVelKernel<<<gridSize,blockSize>>>( (float4*)d_vertexData, massXNum, massYNum );

    cudaErrorCheck( cudaGraphicsUnmapResources( 1, &vboResource, 0 ) );
}

__global__ void updatePosKernel(float4* vertexData, glm::vec3* pos, glm::vec3* vel, glm::vec3* accel, int massXNum, int massYNum )
{
    int2 idx;
    int2 adj;
    int offset;
    glm::vec3 vec;

    idx.x = blockIdx.x * blockDim.x + threadIdx.x;
    idx.y = blockIdx.y * blockDim.y + threadIdx.y;

}