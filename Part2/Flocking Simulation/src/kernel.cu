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
const float boidMass = 3e8;
const __device__ float starMass = 5e10;

const float scene_scale = 2e2; //size of the height map in simulation space
const __device__ float neighborRadius = 8.0f;
const __device__ float g_fVelKv = 0.5f; 
const __device__ float g_fMaxSpeed = 4.0f;
const __device__ float g_fMaxAccel = 10.0f;
const __device__ float neighborAngle = 180.0f;

glm::vec4 * dev_pos;
glm::vec3 * dev_vel;

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
        arr[index].z = 0.0f;//rand.z;
        arr[index].w = mass;
    }
}

//Determine velocity from the distance from the center star. Not super physically accurate because 
//the mass ratio is too close, but it makes for an interesting looking scene
__global__ void generateCircularVelArray(int time, int N, glm::vec3 * arr, glm::vec4 * pos)
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if(index < N)
    {
        glm::vec3 R = glm::vec3(pos[index].x, pos[index].y, pos[index].z);
        float r = glm::length(R) + EPSILON;
        float s = sqrt(G*starMass/r);
        glm::vec3 D = glm::normalize(glm::cross(R/r,glm::vec3(0,0,1)));
        arr[index].x = s*D.x;
        arr[index].y = s*D.y;
        arr[index].z = s*D.z;
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
        arr[index].z = 0.0;//rand.z;
    }
}

//TODO: Determine force between two bodies
__device__ glm::vec3 calculateAcceleration(glm::vec4 us, glm::vec4 them)
{
    //    G*m_us*m_them
    //F = -------------
    //         r^2
    //
    //    G*m_us*m_them   G*m_them
    //a = ------------- = --------
    //      m_us*r^2        r^2
	glm::vec3 usPosition(us.x, us.y, us.z);
	glm::vec3 themPosition(them.x, them.y, them.z);
	glm::vec3 gravityDir = themPosition - usPosition;
	float _1overR = abs(glm::length(gravityDir)) < EPSILON ? 0 : 1.0f / glm::length(gravityDir);

    return gravityDir * (float)G * them.w * _1overR*_1overR*_1overR;
}

//TODO: Core force calc kernel global memory
__device__  glm::vec3 naiveAcc(int N, glm::vec4 my_pos, glm::vec4 * their_pos)
{
    glm::vec3 acc = calculateAcceleration(my_pos, glm::vec4(0,0,0,starMass));
	for(int i = 0; i < N; ++i) // read global memory N times
	{
		acc += calculateAcceleration(my_pos, their_pos[i]);
	}
//	printf("acc.x = %f, acc.y = %f, acc.z = %f\n", acc.x, acc.y, acc.z);
    return acc;
}


//TODO: Core force calc kernel shared memory
__device__ glm::vec3 sharedMemAcc(int N, glm::vec4 my_pos, glm::vec4 * their_pos)
{
    glm::vec3 acc = calculateAcceleration(my_pos, glm::vec4(0,0,0,starMass));

	__shared__ glm::vec4 sharedPositions[blockSize];
//	if((threadIdx.x + (blockIdx.x * blockDim.x)) == 0) printf("%d\n",(int)ceil((float)N / blockSize));
	int positionsFullBlocks = (int)ceil((float)N / blockSize);
	for(int i = 0; i < positionsFullBlocks; ++i)
	{
		int index = threadIdx.x + i * blockDim.x;
/*
		if(index < N)
		{
			sharedPositions[threadIdx.x] = their_pos[index];		
		}
		__syncthreads();
		for(int j = 0; j < blockSize; ++j) 
		{
			acc += calculateAcceleration(my_pos, sharedPositions[j]);
		}
		__syncthreads();*/
		if(index < N)
		{
			sharedPositions[threadIdx.x] = their_pos[index];		
			__syncthreads();
			for(int j = 0; j < blockSize; ++j) 
			{
				acc += calculateAcceleration(my_pos, sharedPositions[j]);
			}
			__syncthreads();
		}
		else
			__syncthreads();
	}
//	printf("acc.x = %f, acc.y = %f, acc.z = %f\n", acc.x, acc.y, acc.z);
    return acc;
}


//Integration 
__global__ void update(int N, float dt, glm::vec4 * pos, glm::vec3 * vel)
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
		glm::vec3 centerOfMas(0.0f);
		// find neighborhood
		for(int i = 0; i < N; ++i) 
		{
			glm::vec3 theirPos(pos[i].x, pos[i].y, pos[i].z);
			float distanceToNeighbor = glm::distance(myPosition, theirPos) + EPSILON;
			if(distanceToNeighbor < neighborRadius && glm::dot(glm::normalize(myVelocity), glm::normalize(theirPos - myPosition)) > cos(neighborAngle/2))
			{
				alignmentNumerator += vel[i];				
				separationVel += (myPosition - theirPos) / distanceToNeighbor /distanceToNeighbor;
				centerOfMas += theirPos;
				++numberOfNeighbors;

			}
		}
		alignmentVelocity = numberOfNeighbors > 0 ? (alignmentNumerator / float(numberOfNeighbors)) : myVelocity;
		centerOfMas = numberOfNeighbors > 0 ? (centerOfMas / float(numberOfNeighbors)) : myPosition;
		glm::vec3 desiredVel = alignmentVelocity + 0.5f*separationVel + 0.05f*(centerOfMas - myPosition);
		// calculate allignment velocity

		// calculate cohesion velocity

		// calculate separation velocity

//        glm::vec3 acc = ACC(N, my_pos, pos);

		vel[index] += g_fVelKv * (desiredVel - myVelocity) * dt;

    }
}

__global__ void updatePosition(int N, float dt, glm::vec4 * pos, glm::vec3 * vel)
{
	int index = threadIdx.x + (blockIdx.x * blockDim.x);
    if( index < N )
    {
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
__global__ void sendToVBO(int N, glm::vec4 * pos, float * vbo, int width, int height, float s_scale)
{
    int index = threadIdx.x + (blockIdx.x * blockDim.x);

    float c_scale_w = -2.0f / s_scale;
    float c_scale_h = -2.0f / s_scale;

    if(index<N)
    {
        vbo[4*index+0] = pos[index].x*c_scale_w;
        vbo[4*index+1] = pos[index].y*c_scale_h;
        vbo[4*index+2] = 0;
        vbo[4*index+3] = 1;
    }
}

//Update the texture pixel buffer object
//(This texture is where openGL pulls the data for the height map)
__global__ void sendToPBO(int N, glm::vec4 * pos, float4 * pbo, int width, int height, float s_scale)
{
    int index = threadIdx.x + (blockIdx.x * blockDim.x);
    int x = index % width;
    int y = index / width;
    float w2 = width / 2.0;
    float h2 = height / 2.0;

    float c_scale_w = width / s_scale;
    float c_scale_h = height / s_scale;

    if(x<width && y<height)
    {
        glm::vec3 color(0.05, 0.15, 0.3);
        glm::vec3 acc = ACC(N, glm::vec4((x-w2)/c_scale_w,(y-h2)/c_scale_h,0,1), pos);
        float mag = sqrt(sqrt(acc.x*acc.x + acc.y*acc.y + acc.z*acc.z));
        // Each thread writes one pixel location in the texture (textel)
        pbo[index].w = (mag < 1.0f) ? mag : 1.0f;
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

    generateRandomPosArray<<<fullBlocksPerGrid, blockSize>>>(1, numObjects, dev_pos, scene_scale, boidMass); // one dimensional block
    checkCUDAErrorWithLine("Kernel failed!");
//    generateCircularVelArray<<<fullBlocksPerGrid, blockSize>>>(2, numObjects, dev_vel, dev_pos);
	generateRandomVelArray<<<fullBlocksPerGrid, blockSize>>>(2, numObjects, dev_vel, scene_scale/100.0);
    checkCUDAErrorWithLine("Kernel failed!");
}

void cudaFlockingUpdateWrapper(float dt)
{
    dim3 fullBlocksPerGrid((int)ceil(float(numObjects)/float(blockSize)));
    update<<<fullBlocksPerGrid, blockSize>>>(numObjects, dt, dev_pos, dev_vel);
    checkCUDAErrorWithLine("Kernel failed!");
	updatePosition<<<fullBlocksPerGrid, blockSize>>>(numObjects, dt, dev_pos, dev_vel);
	checkCUDAErrorWithLine("Kernel failed!");
}

void cudaUpdatePBO(float4 * pbodptr, int width, int height)
{
    dim3 fullBlocksPerGrid((int)ceil(float(width*height)/float(blockSize)));
    sendToPBO<<<fullBlocksPerGrid, blockSize>>>(numObjects, dev_pos, pbodptr, width, height, scene_scale);
    checkCUDAErrorWithLine("Kernel failed!");
}

void cudaUpdateVBO(float * vbodptr, int width, int height)
{
    dim3 fullBlocksPerGrid((int)ceil(float(numObjects)/float(blockSize)));
    sendToVBO<<<fullBlocksPerGrid, blockSize>>>(numObjects, dev_pos, vbodptr, width, height, scene_scale);
    checkCUDAErrorWithLine("Kernel failed!");
}


