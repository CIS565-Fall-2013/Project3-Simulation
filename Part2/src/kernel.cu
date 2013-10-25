#include <stdio.h>
#include <cuda.h>
#include <cmath>
#include "glm/glm.hpp"
#include "utilities.h"
#include "kernel.h"

#if SHARED == 0
    #define ACC(x,y,z) sharedMemAcc(x,y,z)
#else
    #define ACC(x,y,z) naiveAcc(x,y,z)
#endif

//GLOBALS
dim3 threadsPerBlock(blockSize);

int numObjects;

const float scene_scale = 2e2; //size of the height map in simulation space

glm::vec4 * dev_pos;
glm::vec3 * dev_vel;
glm::vec3 * dev_acc;

boid* nBoids;
int iteration = 1;
const __device__ float WALL = 40.0f;			//wall boundary, imaginary cube
const __device__ float NEIGHBOR_RAD = 8.0f;	//radius of neighborhood
const __device__ float MAX_VEL = 3.0f;

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

//Generate randomized starting positions for the planets in the XY plane
//Also initialized the masses
__global__
void generateRandomPosArray(int time, int N, boid* arr, float scale)
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if(index < N)
    {
        glm::vec3 rand = 10.0f*(generateRandomNumberFromThread(time, index)-0.5f);
		arr[index].pos = rand;
		//arr[index].pos.z = 0;
    }
}

//Generate randomized starting velocities in the XY plane
__global__
void generateRandomVelArray(int time, int N, boid* arr, float scale)
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if(index < N)
    {
        glm::vec3 rand = (generateRandomNumberFromThread(time, index) - 0.5f);
		arr[index].vel = rand;
		//arr[index].vel.z = 0;
    }
}

//generate random point on a sphere
__device__ glm::vec3 getRandomPointOnSphere(float randomSeed, int time){
	
	thrust::default_random_engine rng(hash(randomSeed*time));
	thrust::uniform_real_distribution<float> u01(-1, 1);
    thrust::uniform_real_distribution<float> u02(0, 2*PI);

    glm::vec3 point (0);
        
    float z = (float)u01(rng);
    float theta = (float)u02(rng);

    point.x = sqrt(1 - (z*z)) * cos(theta);
    point.y = sqrt( 1 - (z*z)) * sin(theta);
    point.z = z;
	
	return point;
}

__device__
glm::vec3 wander(glm::vec3 vel, glm::vec3 pos, int index, float dt, int time){
	
	//find random displacement on sphere
	glm::vec3 vWander = getRandomPointOnSphere(index, time);
	
	//constrain in semicircle
	if(glm::dot(vWander, vel) < 0.0f)
		vWander *= -1.0f;

	return vWander;
}

__device__
glm::vec3 separation(int N, glm::vec3 myPos, boid* boids, int time){
	
	int index = blockDim.x * blockIdx.x + threadIdx.x;
	int numTiles = ceil((float)N/blockSize);
	
	//array storing position and velocities, pos goes from 0 to blockSize-1, vel from blockSize - 2*blockSize-1
	__shared__ glm::vec3 posTile[blockSize];	
	
	glm::vec3 vSeparation(0);
	int numNeigbors = 0;

	for(int i = 0; i < numTiles; ++i){
		//load positions to shared memory
		posTile[threadIdx.x] = boids[i*blockSize +threadIdx.x].pos;
		__syncthreads();

		//find weighted average of velocities in neighborhood
		for( int j = 0; j < blockSize && (blockSize*i+j)<N; ++j){
			//check if in neigborhood
			glm::vec3 dist = myPos - posTile[j];
			float distLen = glm::length(dist);
			if(distLen < NEIGHBOR_RAD){
				if(distLen > EPSILON)
					vSeparation += dist*1.0f/(glm::dot(dist, dist));
				else{
					vSeparation += generateRandomNumberFromThread(time, index);
					vSeparation -=glm::vec3(0.5);
				}
				++numNeigbors;
			}
		}
		__syncthreads();
	}

	return vSeparation*(1.0f/numNeigbors);
}

__device__
glm::vec3 cohesion(int N, glm::vec3 myPos, boid* boids){

	int numTiles = ceil((float)N/blockSize);
	__shared__ glm::vec3 posTile[blockSize];
	int numNeighbors = 0;
	glm::vec3 cM (0);

	for(int i = 0; i < numTiles; ++i){

		//read into shared mem
		posTile[threadIdx.x] = boids[threadIdx.x+i*blockSize].pos;
		__syncthreads();

		for(int j = 0; j<blockSize && i*blockSize+j<N; ++j){
			float dist = glm::length(myPos - posTile[j]);
			//find center of mass
			if(dist < NEIGHBOR_RAD){
				cM += posTile[j];
				numNeighbors ++;
			}
		}
		__syncthreads();
	}

	cM *= 1.0f/numNeighbors;
	
	return cM-myPos;
}

__device__
glm::vec3 alignment(int N, glm::vec3 myPos, glm::vec3 myVel, boid* boids){
	
	int numTiles = ceil((float)N/blockSize);
	
	//array storing position and velocities, pos goes from 0 to blockSize-1, vel from blockSize - 2*blockSize-1
	__shared__ glm::vec3 posVelTile[2*blockSize];	

	glm::vec3 vAlign(0);
	int numNeigbors = 0;

	for(int i = 0; i < numTiles; ++i){
		//load positions and velocitieds to shared memory
		posVelTile[threadIdx.x] = boids[i*blockSize +threadIdx.x].pos;
		posVelTile[blockSize+threadIdx.x] = boids[i*blockSize +threadIdx.x].vel;
		__syncthreads();

		//find weighted average of velocities in neighborhood
		for( int j = 0; j < blockSize && (blockSize*i+j)<N; ++j){
			//check if in neigborhood
			float dist = glm::length(posVelTile[j] - myPos);
			if(dist < NEIGHBOR_RAD){
				vAlign += posVelTile[j+blockSize];
				++numNeigbors;
			}
		}
		__syncthreads();
	}

	//find and return average velocity
	return vAlign*(1.0f/numNeigbors);
}

__device__
glm::vec3 avoidance(glm::vec3 myPos, glm::vec3 myVel){

	glm::vec3 avoid(0);

	//check whether colliding with wall on next few timesteps
	glm::vec3 futurePos = myPos + 10.0f*myVel;
	glm::vec3 reflectNormal(0);
	
	//find normal to reflect around
	if(futurePos.x > WALL)
		reflectNormal.x = -1.0f;
	else if(futurePos.x < -WALL)
		reflectNormal.x = 1.0f;
	else if(futurePos.y > WALL)
		reflectNormal.y = -1.0f;
	else if(futurePos.y < -WALL)
		reflectNormal.y = 1.0f;
	else if(futurePos.z > WALL)
		reflectNormal.z = -1.0f;
	else if (futurePos.z < -WALL)
		reflectNormal.z = 1.0f;

	if(glm::dot(reflectNormal, reflectNormal) > EPSILON)
		avoid = glm::reflect(myVel, reflectNormal);

	return avoid;
}

__device__
glm::vec3 pullBack(glm::vec3 myPos, glm::vec3 myVel){
	
	if(myPos.x > WALL && myVel.x >0 ||myPos.x < -WALL && myVel.x <0 ||
		myPos.y > WALL && myVel.y >0 ||myPos.y < -WALL && myVel.y <0 ||
		myPos.z > WALL && myVel.z >0 ||myPos.z < -WALL && myVel.z <0 )
		return -myVel;

	return glm::vec3(0);
}

//updates velocity
__global__
void updateF(int N, float dt, boid* boids, int time)
{
    int index = threadIdx.x + (blockIdx.x * blockDim.x);
    glm::vec3 my_pos;
    glm::vec3 my_vel;

	if(index < N){

		my_pos = boids[index].pos;
		my_vel = boids[index].vel;
		
		my_vel += 0.3f*alignment(N, my_pos, my_vel, boids);
		my_vel += 10.0f*separation(N, my_pos, boids, time);
		my_vel += 0.1f*cohesion(N, my_pos, boids);

		my_vel += 0.5f*wander(my_vel, my_pos, index, dt, time);

		my_vel += 0.5f*avoidance(my_pos, my_vel);
		my_vel += 0.5f*pullBack(my_pos, my_vel);
		
		//clamp vel and update
		my_vel= glm::clamp(my_vel, -glm::vec3(MAX_VEL), glm::vec3(MAX_VEL));
		//my_vel.z = 0.0f;
		boids[index].vel = my_vel;
	}
}

//does euler integration to find new position
__global__
void updateS(int N, float dt, boid* boids)
{
    int index = threadIdx.x + (blockIdx.x * blockDim.x);
	glm::vec3 pos;
	glm::vec3 vel;
    if( index < N )
    {
		pos = boids[index].pos;
		vel = boids[index].vel;
        pos.x += vel.x * dt;
        pos.y += vel.y * dt;
        pos.z += vel.z * dt;

		boids[index].pos = pos;

    }
}

//Update the vertex buffer object
//(The VBO is where OpenGL looks for the positions for the planets)
__global__
	void sendToVBO(int N, boid* boids, float * vbo, float* velbo, int width, int height, float s_scale)
{
    int index = threadIdx.x + (blockIdx.x * blockDim.x);

    float c_scale_w = -2.0f / s_scale;
    float c_scale_h = -2.0f / s_scale;
	float c_scale_b = -2.0f / s_scale;

    if(index<N)
    {
		glm::vec3 pos = boids[index].pos;
        vbo[4*index+0] = pos.x*c_scale_w;
        vbo[4*index+1] = pos.y*c_scale_h;
		vbo[4*index+2] = pos.z*c_scale_b;
        vbo[4*index+3] = 1;

		glm::vec3 vel = boids[index].vel;
		velbo[4*index+0] = vel.x;
		velbo[4*index+1] = vel.y;
		velbo[4*index+2] = vel.z;
		velbo[4*index+3] = 0;

    }
}

//Update the texture pixel buffer object
//(This texture is where openGL pulls the data for the height map)
__global__
void sendToPBO(int N, glm::vec4 * pos, float4 * pbo, int width, int height, float s_scale)
{
   // int index = threadIdx.x + (blockIdx.x * blockDim.x);
}

/*************************************
 * Wrappers for the __global__ calls *
 *************************************/

//Initialize memory, update some globals
void initCuda(int N)
{
    numObjects = N;
    dim3 fullBlocksPerGrid((int)ceil(float(N)/float(blockSize)));

    //cudaMalloc((void**)&dev_pos, N*sizeof(glm::vec4));
    //checkCUDAErrorWithLine("Kernel failed!");
    //cudaMalloc((void**)&dev_vel, N*sizeof(glm::vec3));
    //checkCUDAErrorWithLine("Kernel failed!");
    //cudaMalloc((void**)&dev_acc, N*sizeof(glm::vec3)); 
    //checkCUDAErrorWithLine("Kernel failed!");

	cudaMalloc((void**)&nBoids, N*sizeof(boid));
	checkCUDAErrorWithLine("Kernel failed!");

	generateRandomPosArray<<<fullBlocksPerGrid, blockSize>>>(1, numObjects, nBoids, scene_scale);
    checkCUDAErrorWithLine("Kernel failed!");

	generateRandomVelArray<<<fullBlocksPerGrid, blockSize>>>(2, numObjects, nBoids, scene_scale);
	checkCUDAErrorWithLine("Kernel failed!");

    cudaThreadSynchronize();
}

void cudaNBodyUpdateWrapper(float dt)
{
    dim3 fullBlocksPerGrid((int)ceil(float(numObjects)/float(blockSize)));
	updateF<<<fullBlocksPerGrid, blockSize, blockSize*sizeof(boid)>>>(numObjects, dt, nBoids, iteration);
    checkCUDAErrorWithLine("Kernel failed!");
	updateS<<<fullBlocksPerGrid, blockSize>>>(numObjects, dt, nBoids);
    checkCUDAErrorWithLine("Kernel failed!");
    cudaThreadSynchronize();
}

void cudaUpdateVBO(float * vbodptr, float* velboptr, int width, int height)
{
    dim3 fullBlocksPerGrid((int)ceil(float(numObjects)/float(blockSize)));
	sendToVBO<<<fullBlocksPerGrid, blockSize>>>(numObjects, nBoids, vbodptr, velboptr, width, height, scene_scale);
    cudaThreadSynchronize();
	iteration ++;
}

void cudaUpdatePBO(float4 * pbodptr, int width, int height)
{
    dim3 fullBlocksPerGrid((int)ceil(float(width*height)/float(blockSize)));
    sendToPBO<<<fullBlocksPerGrid, blockSize, blockSize*sizeof(glm::vec4)>>>(numObjects, dev_pos, pbodptr, width, height, scene_scale);
    cudaThreadSynchronize();
}
