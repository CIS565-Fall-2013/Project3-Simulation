#include <stdio.h>
#include <cuda.h>
#include <cmath>
#include "glm/glm.hpp"
#include "utilities.h"
#include "kernel.h"

#if SHARED == 1
    #define ACC(x,y,z) sharedMemAcc(x,y,z)
#else
    #define ACC(x,y,z) naiveAcc(x,y,z)
#endif

//GLOBALS
dim3 threadsPerBlock(blockSize);

int numObjects;
const float boidMass = 1.0f;
const float scene_scale = 4e2; 

const __device__ float neighborRadius = 20.0f;
const __device__ float g_fMaxForce = 1.0f;
const __device__ float neighborAngle = 180.0f;
const __device__ float c_alignment = 2.0f;
const __device__ float c_separation = 3.0f;
const __device__ float c_cohesion = 0.005f;
const __device__ float c_seek = 0.001f;


glm::vec4 * dev_pos;
glm::vec3 * dev_vel;
glm::vec3 * dev_acc; 

void checkCUDAError(const char *msg, int line = -1)
{
    cudaError_t err = cudaGetLastError();
    if( cudaSuccess != err)
    {
        if( line >= 0 )
        {
            fprintf(stderr, "Line %d: ", line);
        }
        fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString( err) ); 
		getchar();
        exit(EXIT_FAILURE); 
    }
} 

__host__ __device__ unsigned int hash(unsigned int a){
    a = (a+0x7ed55d16) + (a<<12);
    a = (a^0xc761c23c) ^ (a>>19);
    a = (a+0x165667b1) + (a<<5);
    a = (a+0xd3a2646c) ^ (a<<9);
    a = (a+0xfd7046c5) + (a<<3);
    a = (a^0xb55a4f09) ^ (a>>16);
    return a;
}

__host__ __device__ glm::vec3 truncate(glm::vec3 direction, float maxLength)
{
	if(glm::length(direction) > maxLength)
		return glm::normalize(direction) * maxLength;
	else
		return direction;
}

//Function that generates static.
__host__ __device__ glm::vec3 generateRandomNumberFromThread(float time, int index)
{
    thrust::default_random_engine rng(hash(index*time));
    thrust::uniform_real_distribution<float> u01(0,1);

    return glm::vec3((float) u01(rng), (float) u01(rng), (float) u01(rng));
}

//Generate randomized starting positions for the planets in the XY plane
//Also initialized the masses
__global__ void generateRandomPosArray(int time, int N, glm::vec4 * arr, float scale, float mass)
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if(index < N)
    {
        glm::vec3 rand = scale*(generateRandomNumberFromThread(time, index)-0.5f);
        arr[index].x = rand.x;
        arr[index].y = rand.y;
        arr[index].z = /*0.0f;*/rand.z;
        arr[index].w = mass;
    }
}

//Generate randomized starting velocities in the XY plane
__global__ void generateRandomVelArray(int time, int N, glm::vec3 * arr, float scale)
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if(index < N)
    {
        glm::vec3 rand = scale*(generateRandomNumberFromThread(time, index) - 0.5f);
        arr[index].x = rand.x;
        arr[index].y = rand.y;
        arr[index].z = /*0.0;*/rand.z;
    }
}


//Integration 
__global__ void updateAccelaration(int N, float dt, glm::vec4 *pos, glm::vec3 *vel, glm::vec3 *acc, glm::vec3 target)
{
    int index = threadIdx.x + (blockIdx.x * blockDim.x);
    if( index < N )
    {
		glm::vec3 myPosition(pos[index].x, pos[index].y, pos[index].z);
		glm::vec3 myVelocity = vel[index];

		int numberOfNeighbors = 0;
		glm::vec3 alignmentNumerator(0.0f);	
		glm::vec3 alignmentVelocity(0.0f);
		glm::vec3 separationVel(0.0f);
		glm::vec3 centerOfMass(0.0f);
		glm::vec3 desiredVel(0.0f);
		// Calculate desired velocity
		for(int i = 0; i < N; ++i) 
		{
			glm::vec3 theirPos(pos[i].x, pos[i].y, pos[i].z);
			float distanceToNeighbor = glm::distance(myPosition, theirPos) + EPSILON;
			if(distanceToNeighbor < neighborRadius && glm::dot(glm::normalize(myVelocity), glm::normalize(theirPos - myPosition)) > cos(neighborAngle/2))
			{
				alignmentNumerator += vel[i];				
				separationVel += (myPosition - theirPos) / distanceToNeighbor /distanceToNeighbor;
				centerOfMass += theirPos;
				++numberOfNeighbors;
			}
		}
		if(numberOfNeighbors > 0)
		{
			alignmentVelocity = alignmentNumerator / float(numberOfNeighbors);
			centerOfMass = centerOfMass / float(numberOfNeighbors);
			desiredVel = c_alignment*alignmentVelocity + c_separation*separationVel + c_cohesion*(centerOfMass - myPosition) + c_seek * glm::normalize(target-myPosition);
		}
		else desiredVel = c_seek * (target-myPosition);

		if(glm::length(myPosition) > 800.0f) desiredVel = glm::normalize(-myPosition);
		
		// Calculate acceleration from steering direction
		acc[index] = truncate(desiredVel - myVelocity, g_fMaxForce) / pos[index].w;

    }
}

__global__ void updatePosition(int N, float dt, glm::vec4 *pos, glm::vec3 *vel, glm::vec3 *acc)
{
	int index = threadIdx.x + (blockIdx.x * blockDim.x);
    if( index < N )
    {
		//vel[index] = truncate(vel[index] + acc[index] * dt, g_fMaxSpeed);
		vel[index] = glm::normalize(vel[index] + acc[index] * dt) * g_fMaxSpeed;
        // RK4 method
		glm::vec3 k1 = vel[index];
		glm::vec3 k2 = k1 + 0.5f * dt * k1;
		glm::vec3 k3 = k1 + 0.5f * dt * k2;
		glm::vec3 k4 = k1 + dt * k3;

		glm::vec3 increment = 1.0f/6.0f * (k1 + 2.0f*k2 + 2.0f*k3 + k4);

		pos[index].x += increment.x * dt;
        pos[index].y += increment.y * dt;
        pos[index].z += increment.z * dt;

		 //Euler method
        /*pos[index].x += vel[index].x * dt;
        pos[index].y += vel[index].y * dt;
        pos[index].z += vel[index].z * dt;*/
    }
}

//Update the vertex buffer object
//(The VBO is where OpenGL looks for the positions for the planets)
__global__ void sendToVBO(int N, glm::vec4 * pos, glm::vec3 *vel, float * posVBO, float *velVBO, float s_scale)
{
    int index = threadIdx.x + (blockIdx.x * blockDim.x);

    float c_scale_w = 2.0f / s_scale;
    float c_scale_h = 2.0f / s_scale;
	float c_scale_s = 2.0f / s_scale;

    if(index<N)
    {
        posVBO[4*index+0] = pos[index].x*c_scale_w;
        posVBO[4*index+1] = pos[index].y*c_scale_h;
        posVBO[4*index+2] = pos[index].z*c_scale_s;
        posVBO[4*index+3] = 1;

		velVBO[3*index+0] = vel[index].x*c_scale_w;
		velVBO[3*index+1] = vel[index].y*c_scale_h;
		velVBO[3*index+2] = vel[index].z*c_scale_s;
    }
}

/*************************************
 * Wrappers for the __global__ calls *
 *************************************/

//Initialize memory, update some globals
void initCuda(int N)
{
    numObjects = N;
    dim3 fullBlocksPerGrid((int)ceil(float(N)/float(blockSize))); // one dimensional grid

    cudaMalloc((void**)&dev_pos, N*sizeof(glm::vec4));
    checkCUDAErrorWithLine("Kernel failed!");
    cudaMalloc((void**)&dev_vel, N*sizeof(glm::vec3));
    checkCUDAErrorWithLine("Kernel failed!");
	cudaMalloc((void**)&dev_acc, N*sizeof(glm::vec3));  
    checkCUDAErrorWithLine("Kernel failed!");  

    generateRandomPosArray<<<fullBlocksPerGrid, blockSize>>>(1, numObjects, dev_pos, scene_scale, boidMass); // one dimensional block
    checkCUDAErrorWithLine("Kernel failed!");
	generateRandomVelArray<<<fullBlocksPerGrid, blockSize>>>(2, numObjects, dev_vel, g_fMaxSpeed);
    checkCUDAErrorWithLine("Kernel failed!");
}

void cudaFlockingUpdateWrapper(float dt, glm::vec3 target)
{
    dim3 fullBlocksPerGrid((int)ceil(float(numObjects)/float(blockSize)));
    updateAccelaration<<<fullBlocksPerGrid, blockSize>>>(numObjects, dt, dev_pos, dev_vel, dev_acc, target);
    checkCUDAErrorWithLine("Kernel failed!");
	updatePosition<<<fullBlocksPerGrid, blockSize>>>(numObjects, dt, dev_pos, dev_vel, dev_acc);
	checkCUDAErrorWithLine("Kernel failed!");
}

void cudaUpdateVBO(float *vbodptr, float *velptr)
{
    dim3 fullBlocksPerGrid((int)ceil(float(numObjects)/float(blockSize)));
    sendToVBO<<<fullBlocksPerGrid, blockSize>>>(numObjects, dev_pos, dev_vel, vbodptr, velptr, scene_scale);
    checkCUDAErrorWithLine("Kernel failed!");
}


