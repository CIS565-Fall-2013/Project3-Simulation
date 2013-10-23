#include <stdio.h>
#include "util.h"
#include "kernel.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"


#define MASS .5f
#define SHEAR_LEN 1.414213562f
#define BEND_LEN 2.0f
#define DAMPING 1.0f
#define G 2.8f
#define K 200.0f

float* d_vertexData = 0;
float* d_normal = 0;

glm::vec3* d_vel = 0;
glm::vec3* d_force = 0;
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
    cudaErrorCheck( cudaMalloc((void**)&d_force, massXNum * massYNum * sizeof( glm::vec3 ) ) );
    cudaErrorCheck( cudaMalloc((void**)&d_pos, massXNum * massYNum * sizeof( glm::vec3 ) ) );
    cudaErrorCheck( cudaMalloc((void**)&d_pos2, massXNum * massYNum * sizeof( glm::vec3 ) ) );

    cudaErrorCheck( cudaMemset( (void*)d_vel, 0, massXNum * massYNum * sizeof( glm::vec3 ) ) );
    cudaErrorCheck( cudaMemset( (void*)d_force, 0, massXNum * massYNum * sizeof( glm::vec3 ) ) );

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

    if( d_force )
        cudaFree( d_force );
    d_force = 0;

    if( d_pos )
        cudaFree( d_pos );
    d_pos = 0;

    if( d_pos2 )
        cudaFree( d_pos2 );
    d_pos2 = 0;

    return 0;
}

__global__ void updateVelKernel( float4* vertexData, glm::vec3* pos, glm::vec3* force, glm::vec3* vel,
                                 float2 restlen, int massXNum, int massYNum )
{
    int2 idx;
    int2 adj;
    int offset;
    glm::vec3 vec;
    idx.x = blockIdx.x * blockDim.x + threadIdx.x;
    idx.y = blockIdx.y * blockDim.y + threadIdx.y;
    offset = idx.y * massXNum + idx.x;
    glm::vec3 internalforce( 0.0f,0.0f,0.0f );

    if( idx.x < massXNum && idx.y < massYNum )
    {
        //internal force - stretch
        adj.x = idx.x-1;
        if( (adj.x) >= 0 )
        {
            vec = pos[ offset - 1] - pos[ offset ];
            internalforce += K * ( vec - restlen.x * glm::normalize( vec ) ) ;

        }
        adj.x = idx.x+1;
        if( ( adj.x ) < massXNum )
        {
            vec = pos[ idx.y * massXNum + adj.x] - pos[ offset ];
            internalforce += K * ( vec - restlen.x * glm::normalize( vec ) );
        }
        adj.y = idx.y - 1;
        if( (adj.y) >= 0 )
        {
            vec = pos[ adj.y * massXNum + idx.x] - pos[ offset ];
            internalforce += K * ( vec - restlen.x * glm::normalize( vec ) );
        }
        adj.y = idx.y+1;
        if( ( adj.y) < massYNum )
        {
            vec = pos[ adj.y * massXNum + idx.x] - pos[ offset ];
            internalforce += K * ( vec - restlen.x * glm::normalize( vec ) );
        }

        //internal force - shear
        if( (adj.x = idx.x -1 ) >= 0 && (adj.y=idx.y-1) >=0 )
        {
            vec = pos[ adj.y * massXNum + adj.x] - pos[ offset ];
            internalforce += K * ( vec - SHEAR_LEN*restlen.x * glm::normalize( vec ) );
        }

        if( (adj.x = idx.x +1 ) < massXNum && (adj.y=idx.y+1) < massYNum )
        {
            vec = pos[ adj.y * massXNum + adj.x] - pos[ offset ];
            internalforce += K * ( vec - SHEAR_LEN*restlen.x * glm::normalize( vec ) );
        }

        if( (adj.x = idx.x +1 ) < massXNum && (adj.y=idx.y-1) >= 0)
        {
            vec = pos[ adj.y * massXNum + adj.x] - pos[ offset ];
            internalforce += K * ( vec - SHEAR_LEN*restlen.x * glm::normalize( vec ) );
        }

        if( (adj.x = idx.x -1 ) >= 0 && (adj.y=idx.y+1) < massYNum )
        {
            vec = pos[ adj.y * massXNum + adj.x] - pos[ offset ];
            internalforce += K * ( vec - SHEAR_LEN*restlen.x * glm::normalize( vec ) );
        }

        //if( idx.x == 0 || idx.x == massXNum-1 || idx.y == 0 || idx.y == massYNum-1 )
        //{
            //internal force - bend
            if( (adj.x = idx.x-2) >= 0 )
            {
                vec = pos[ idx.y * massXNum + adj.x] - pos[ offset ];
                internalforce += K * ( vec - BEND_LEN*restlen.x * glm::normalize( vec ) );
            }
            if( ( adj.x = idx.x+2 ) < massXNum )
            {
                vec = pos[ idx.y * massXNum + adj.x] - pos[ offset ];
                internalforce += K * ( vec - BEND_LEN*restlen.x * glm::normalize( vec ) );
            }
            if( (adj.y = idx.y-2) >= 0 )
            {
                vec = pos[ adj.y * massXNum + idx.x] - pos[ offset ];
                internalforce += K * ( vec - BEND_LEN*restlen.x * glm::normalize( vec ) );
            }
            if( ( adj.y = idx.y+2 ) < massYNum )
            {
                vec = pos[ adj.y * massXNum + idx.x] - pos[ offset ];
                internalforce += K * ( vec - BEND_LEN*restlen.x * glm::normalize( vec ) );
            }
       // }
        force[offset] = internalforce;
        
    }
}

void updateVelWrapper( cudaGraphicsResource* &vboResource, float2 restlen, int massXNum, int massYNum )
{
    size_t vboSize;

    dim3 blockSize = dim3(16*16);
    dim3 gridSize = dim3( (massXNum + blockSize.x-1)/blockSize.x, (massYNum + blockSize.y-1)/blockSize.y );

    cudaErrorCheck( cudaGraphicsMapResources( 1, &vboResource, 0 ) );
    cudaErrorCheck( cudaGraphicsResourceGetMappedPointer((void**) &d_vertexData, &vboSize, vboResource ) );

    updateVelKernel<<<gridSize,blockSize>>>( (float4*)d_vertexData, d_pos, d_force, d_vel, restlen, massXNum, massYNum );

    cudaErrorCheck( cudaGraphicsUnmapResources( 1, &vboResource, 0 ) );
}

__global__ void updatePosKernel( float4* vertexData, glm::vec3* pos, glm::vec3* pos2, glm::vec3* accel, float3* normals, 
                                 float dt, float elapse, float windFactor, float2 restlen, int massXNum, int massYNum,
                                 glm::vec3 sphere, float radius)
{
    int2 idx;
    int offset;
    float diff;
    glm::vec3 newPos;
    glm::vec3 totalForce;
    glm::vec3 windForce;
    idx.x = blockIdx.x * blockDim.x + threadIdx.x;
    idx.y = blockIdx.y * blockDim.y + threadIdx.y;
    offset = idx.y * massXNum + idx.x;
    glm::vec3 N;
    if( idx.x < massXNum && idx.y < massYNum )
    {
        if( idx.y == 0 )
            return;

        //add external forces
        //Gravity
        totalForce = accel[offset] + MASS * glm::vec3( 0, -G, 0 );

        //Damping force
        totalForce -= DAMPING*(pos[offset] - pos2[offset])/dt;

        //wind
        //totalForce += glm::vec3( sinf(idx.x*idx.y*elapse),cosf(0*elapse),   );
        N.x = normals[offset].x;
        N.y = normals[offset].y;
        N.z = normals[offset].z;
        windForce =  N *
            glm::dot( N, glm::vec3( 0.0f , 0.0f, sinf(cosf(5.0f*idx.x*idx.y*0*elapse) ) ) ) * windFactor ;
        totalForce += windForce;
        //Verlet Integration
        newPos = 2.0f*pos[offset] - pos2[offset] + (dt*dt*totalForce/MASS);

        //apply stretch constraints
        //if( idx.y > 1 )
        //{
        //    diff = glm::distance( newPos, pos[ (idx.y-1)*(massXNum)+idx.x] ) - 1.1f * restlen.x;
        //    if( glm::distance( newPos, pos[(idx.y-1)*(massXNum)+idx.x] ) > 1.1f * restlen.x )
        //        newPos =  pos[(idx.y-1)*massXNum+idx.x] + glm::normalize( newPos - pos[(idx.y-1)*massXNum+idx.x] ) * 1.1f * restlen.x;
        //}
        //if( idx.x > 1 )
        //{
        //    if( glm::distance( newPos, pos[idx.y*massXNum+idx.x-1] ) > 1.1f * restlen.x )
        //        newPos =  pos[idx.y*massXNum+idx.x-1] + glm::normalize( newPos - pos[idx.y*massXNum+idx.x-1] ) * 1.1f * restlen.x;
        //}
        //if( idx.y < massYNum - 1 )
        //{
        //    if( glm::distance( newPos, pos[(idx.y+1)*(massXNum) + idx.x] ) > 1.1f * restlen.x )
        //        newPos = pos[(idx.y+1)*(massXNum)  + idx.x] +  glm::normalize( newPos - pos[(idx.y+1)*(massXNum) + idx.x] ) * 1.1f * restlen.x;
        //}
        //if( idx.x < massXNum - 1 )
        //{
        //    if( glm::distance( newPos, pos[idx.y*massXNum+idx.x+1] ) > 1.1f * restlen.x )
        //        newPos = pos[idx.y*massXNum+idx.x+1] + glm::normalize( newPos - pos[idx.y*massXNum+idx.x+1] ) * 1.1f * restlen.x;
        //}

        if( glm::distance( newPos, sphere ) < radius+0.05f )
            newPos = sphere + glm::normalize( newPos-sphere) * (radius+0.05f);

        pos2[offset] = pos[offset];
        pos[offset] = newPos;
    }
    
    syncthreads();
    if(idx.x < massXNum && idx.y < massYNum)
    {
        
        vertexData[ offset ].x = newPos.x;
        vertexData[ offset ].y = newPos.y;
        vertexData[ offset ].z = newPos.z;

    }
    
  
}

__global__ void updateInterPosKernel( float4* vertexData, glm::vec3* pos, int massXNum, int massYNum )
{
    int2 idx;
    int offset;
    idx.x = blockIdx.x * blockDim.x + threadIdx.x;
    idx.y = blockIdx.y * blockDim.y + threadIdx.y;
    offset = idx.y * massXNum + idx.x;

    if( idx.x < massXNum && idx.y < massYNum )
    {
        pos[offset].x = vertexData[offset].x;
        pos[offset].y = vertexData[offset].y;
        pos[offset].z = vertexData[offset].z;
    }
}

void updatePosWrapper( cudaGraphicsResource* &vboResource, cudaGraphicsResource* normalVboSrc,
                       float dt, float elapse, float windFactor,
                       float2 restlen, int massXNum, int massYNum, glm::vec3 &sphere, float radius )
{
    size_t vboSize;

    dim3 blockSize = dim3(16*16);
    dim3 gridSize = dim3( (massXNum + blockSize.x-1)/blockSize.x, (massYNum + blockSize.y-1)/blockSize.y );

    cudaErrorCheck( cudaGraphicsMapResources( 1, &vboResource, 0 ) );
    cudaErrorCheck( cudaGraphicsResourceGetMappedPointer((void**) &d_vertexData, &vboSize, vboResource ) );

    cudaErrorCheck( cudaGraphicsMapResources( 1, &normalVboSrc, 0 ) );
    cudaErrorCheck( cudaGraphicsResourceGetMappedPointer((void**) &d_normal, &vboSize, normalVboSrc ) );

    updatePosKernel<<<gridSize,blockSize>>>( (float4*)d_vertexData,d_pos, d_pos2, d_force, (float3*)d_normal, dt, elapse, 
                                               windFactor, restlen, massXNum, massYNum, sphere, radius );
    updateInterPosKernel<<<gridSize,blockSize>>>( (float4*)d_vertexData, d_pos, massXNum, massYNum );

    cudaErrorCheck( cudaGraphicsUnmapResources( 1, &vboResource, 0 ) );
    cudaErrorCheck( cudaGraphicsUnmapResources( 1, &normalVboSrc, 0 ) );
}

__global__ void updateNormalKernel( float3* normals, glm::vec3* pos, int massXNum, int massYNum )
{
    int2 idx;
    int offset;
    glm::vec3 normal(0.0f, 0.0f, 1.0f );
    idx.x = blockIdx.x * blockDim.x + threadIdx.x;
    idx.y = blockIdx.y * blockDim.y + threadIdx.y;
    offset = idx.y * massXNum + idx.x;

    if( idx.x < massXNum-1 && idx.y < massYNum-1 )
    {
        normal = glm::cross( pos[offset+ massXNum] - pos[offset], pos[offset+1] - pos[offset]  );
        normal = glm::normalize( normal );
    }
    else if( idx.x == massXNum - 1 && idx.y < massYNum-1 )
    {
        normal = glm::cross( pos[offset-1]-pos[offset] , pos[offset+ massXNum ] - pos[offset] );
        normal = glm::normalize( normal );
    }
    else if( idx.y == massYNum - 1 && idx.x< massXNum-1 )
    {
        normal = glm::cross( pos[offset+1] - pos[offset], pos[offset- massXNum ] - pos[offset] );
        normal = glm::normalize( normal );
    }
    else if( idx.x == massXNum-1 && idx.y == massYNum-1 )
    {
        normal = glm::cross( pos[offset-massXNum]-pos[offset], pos[offset- 1 ]- pos[offset] );
        normal = glm::normalize( normal );
    }

    if( idx.x <  massXNum && idx.y < massYNum )
    {
     normals[offset].x = normal.x;
     normals[offset].y = normal.y;
     normals[offset].z = normal.z;
    }
}

void updateNormalWrapper( cudaGraphicsResource* &normalVboSrc, int massXNum, int massYNum )
{
    size_t vboSize;

    dim3 blockSize = dim3(16*16);
    dim3 gridSize = dim3( (massXNum + blockSize.x-1)/blockSize.x, (massYNum + blockSize.y-1)/blockSize.y );

    cudaErrorCheck( cudaGraphicsMapResources( 1, &normalVboSrc, 0 ) );
    cudaErrorCheck( cudaGraphicsResourceGetMappedPointer((void**) &d_normal, &vboSize, normalVboSrc ) );

    updateNormalKernel<<< gridSize, blockSize >>>( (float3*)d_normal, d_pos, massXNum, massYNum );
    cudaErrorCheck( cudaGraphicsUnmapResources( 1, &normalVboSrc, 0 ) );
}