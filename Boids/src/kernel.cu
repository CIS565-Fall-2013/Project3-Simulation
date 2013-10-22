#include <stdio.h>
#include <cuda.h>
#include <cmath>
#include "glm/glm.hpp"
#include "utilities.h"
#include "kernel.h"


//GLOBALS
dim3 threadsPerBlock(blockSize);

int numObjects;

BoidProps * dev_boids;
glm::vec4 * dev_netForces;


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
		system("pause");
		exit(EXIT_FAILURE); 
	}
} 

__host__ __device__
	unsigned int hash(unsigned int a){
		a = (a+0x7ed55d16) + (a<<12);
		a = (a^0xc761c23c) ^ (a>>19);
		a = (a+0x165667b1) + (a<<5);
		a = (a+0xd3a2646c) ^ (a<<9);
		a = (a+0xfd7046c5) + (a<<3);
		a = (a^0xb55a4f09) ^ (a>>16);
		return a;
}

//Function that generates static.
__host__ __device__ 
	glm::vec3 generateRandomNumberFromThread(float time, int index)
{
	thrust::default_random_engine rng(hash(index*time));
	thrust::uniform_real_distribution<float> u01(0,1);

	return glm::vec3((float) u01(rng), (float) u01(rng), (float) u01(rng));
}

//Generate randomized flock (in range (-mapDims.x/2:mapDims.x/2, -mapDims.y/2:mapDims.y/2, 0:mapDims.z))
__global__
	void generateRandomFlock(int time, int N, BoidProps * boids, WorldProps properties)
{
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	if(index < N)
	{
		
		glm::vec3 randPos = properties.InitialDims*(generateRandomNumberFromThread(time, index)-glm::vec3(0.5f,0.5f,0.0f));
		boids[index].pos.x = randPos.x;
		boids[index].pos.y = randPos.y;
		boids[index].pos.z = randPos.z;

		glm::vec3 velocity = properties.InitialMaxVel*generateRandomNumberFromThread(time+1, index);
		boids[index].vel.x = velocity.x;
		boids[index].vel.y = velocity.y;
		boids[index].vel.z = velocity.z;

		boids[index].rollAngle = 0.0;
	}
}


__device__ 
	glm::vec4 applyIndividualRules(WorldProps world, BoidProps me)
{
	//TODO: Apply rules
	return glm::vec4(0,0,0.01,0);
}

__device__ 
	glm::vec4 applyPairwiseRules(WorldProps world, BoidProps me, BoidProps them)
{
	//TODO: Apply rules
	return glm::vec4(0,0,0,0);
}


__device__
	glm::vec4 computeNetForce(int N, WorldProps world, int myIndex, BoidProps me, BoidProps* boids)
{
	extern __shared__ BoidProps shBoids[]; 
	glm::vec4 netForce = glm::vec4(0,0,0,0);//Use 4th field for roll torque

	//Compute for each tile
	int numTiles = N/blockDim.x;
	if(N % blockDim.x != 0)
		numTiles++;//Add a tile for the extras

	//For each full tile
	for(int tile = 0; tile < numTiles; ++tile){

		int tileOffset = tile*blockDim.x;
		//Load into shared memory using coallesed acces
		int loadIndex = threadIdx.x + tileOffset;
		if(loadIndex < N)
			shBoids[threadIdx.x] = boids[loadIndex];

		//Wait for load to finish
		__syncthreads();

		//Perform update for entire tile using shared memory
		//No bank conflicts because this is broadcast
		for(int i = 0; i < blockDim.x; ++i)
		{
			int idx = tileOffset+i;
			if(idx < N && idx != myIndex)//Don't process 
				netForce += applyPairwiseRules(world, me, shBoids[i]);
			else
				break;
		}

		__syncthreads();

	}

	netForce += applyIndividualRules(world, me);
	return netForce;
}

//Simple Euler integration scheme. Update desired velocities
__global__
	void updateForces(int N, WorldProps world, float dt, BoidProps* boids, glm::vec4* netForces)
{ 

	int index = threadIdx.x + (blockIdx.x * blockDim.x);

	if(index < N) 
	{
		netForces[index] = computeNetForce(N, world, index, boids[index], boids);
	}
}

__global__
	void updateS(int N, WorldProps world, float dt, BoidProps* boids, glm::vec4 * netForces)
{
	int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if( index < N )
	{
		//TODO: Apply forces to Boids
		//TODO: Encorporate mass
		//TODO: Include gravity component
		//TODO: Include non-holonomic controller
		boids[index].vel += glm::vec3(netForces[index])/1.0f*dt;//a = F/m
		boids[index].pos += boids[index].vel*dt;
	}
}

//Update the vertex buffer object
//(The VBO is where OpenGL looks for the positions for the planets)
__global__
	void sendToVBO(int N, BoidProps * boids, float * vbo)
{
	int index = threadIdx.x + (blockIdx.x * blockDim.x);

	if(index<N)
	{

		BoidProps me = boids[index];
		//Position
		vbo[boidVBO_PositionOffset + boidVBOStride*index + 0] =  me.pos.x;
		vbo[boidVBO_PositionOffset + boidVBOStride*index + 1] =  me.pos.y;
		vbo[boidVBO_PositionOffset + boidVBOStride*index + 2] =  me.pos.z;
		vbo[boidVBO_PositionOffset + boidVBOStride*index + 3] =  1.0f;

		//TODO: Allow Roll
		glm::vec3 Right = glm::normalize(glm::cross(me.vel, glm::vec3(0,0,1)));

		//Up
		glm::vec3 Up = glm::normalize(glm::cross(Right, me.vel));

		vbo[boidVBO_UpOffset + boidVBOStride*index + 0] = Up.x;
		vbo[boidVBO_UpOffset + boidVBOStride*index + 1] = Up.y;
		vbo[boidVBO_UpOffset + boidVBOStride*index + 2] = Up.z;

		//Forward
		glm::vec3 forward =  glm::normalize(me.vel);
		vbo[boidVBO_ForwardOffset + boidVBOStride*index + 0] = forward.x;
		vbo[boidVBO_ForwardOffset + boidVBOStride*index + 1] = forward.y;
		vbo[boidVBO_ForwardOffset + boidVBOStride*index + 2] = forward.z;

		//Color
		vbo[boidVBO_ColorOffset + boidVBOStride*index + 0] = 1.0f;
		vbo[boidVBO_ColorOffset + boidVBOStride*index + 1] = 1.0f;
		vbo[boidVBO_ColorOffset + boidVBOStride*index + 2] = 1.0f;

		//Shape
		vbo[boidVBO_ShapeOffset + boidVBOStride*index + 0] = 0.1f;//Length
		vbo[boidVBO_ShapeOffset + boidVBOStride*index + 1] = 0.1f;//Wingspan
		vbo[boidVBO_ShapeOffset + boidVBOStride*index + 2] = 0.05f;//Delta
		vbo[boidVBO_ShapeOffset + boidVBOStride*index + 3] = 0.0f;//Wing Deflection (degrees)
	}
}


/*************************************
* Wrappers for the __global__ calls *
*************************************/

//Initialize memory, update some globals
void initCuda(int N, WorldProps properties)
{
	numObjects = N;
	dim3 fullBlocksPerGrid((int)ceil(float(N)/float(blockSize)));

	//Initialize status arrays
	cudaMalloc((void**)&dev_boids, N*sizeof(BoidProps));
	checkCUDAErrorWithLine("Kernel failed Init()!");
	cudaMalloc((void**)&dev_netForces, N*sizeof(glm::vec4));
	checkCUDAErrorWithLine("Kernel failed Init()!");

	generateRandomFlock<<<fullBlocksPerGrid, blockSize>>>(1, numObjects, dev_boids, properties);
	checkCUDAErrorWithLine("Kernel failed!");
	cudaThreadSynchronize();
}

void cudaNBodyUpdateWrapper(float dt, WorldProps worldProps)
{
	dim3 fullBlocksPerGrid((int)ceil(float(numObjects)/float(blockSize)));
	updateForces<<<fullBlocksPerGrid, blockSize, blockSize*sizeof(BoidProps)>>>(numObjects, worldProps, dt, dev_boids, dev_netForces);
	checkCUDAErrorWithLine("Kernel Update failed!");
	updateS<<<fullBlocksPerGrid, blockSize>>>(numObjects, worldProps, dt, dev_boids, dev_netForces);
	checkCUDAErrorWithLine("Kernel failed!");
	cudaThreadSynchronize();
}

void cudaUpdateVBO(float * vbodptr)
{
	dim3 fullBlocksPerGrid((int)ceil(float(numObjects)/float(blockSize)));
	sendToVBO<<<fullBlocksPerGrid, blockSize>>>(numObjects, dev_boids, vbodptr);
	cudaThreadSynchronize();
}


