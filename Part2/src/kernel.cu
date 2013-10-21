#include <stdio.h>
#include <cuda.h>
#include <cmath>
#include "glm/glm.hpp"
#include "utilities.h"
#include "kernel.h"
#include "simStructs.h"

#if SHARED == 1
    #define ACC(x,y,z) sharedMemAcc(x,y,z)
#else
    #define ACC(x,y,z) naiveAcc(x,y,z)
#endif

#if EULER == 1
	#define INTEGRATE(x,y,z) eulerIntegrate(x,y,z)
#else 
	#define INTEGRATE(x,y,z) RK4Integrate(x,y,z)
#endif

//GLOBALS
dim3 threadsPerBlock(blockSize);

int numObjects;
__device__ int timeStep = 0;

const float scene_scale = WALL_DEPTH; //size of the height map in simulation space

boid* dev_boids;
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
void generateRandomPosArray(int time, int N, boid * arr, float scale)
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if(index < N)
    {
        glm::vec3 rand = (scale - BUFFER)*(generateRandomNumberFromThread(time, index)-0.5f);
        arr[index].pos = rand;
		arr[index].r = 7.0f;
    }
}

//Determine velocity from the distance from the center star. Not super physically accurate because 
//the mass ratio is too close, but it makes for an interesting looking scene
__global__
void generateCircularVelArray(int time, int N, boid * boids)
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if(index < N)
    {
		thrust::default_random_engine rng(hash((time + index) * N * index));
		thrust::uniform_real_distribution<float> u01(MIN_VELOCITY, MAX_VELOCITY);
		thrust::uniform_real_distribution<float> u02(-PI, PI);

		float theta = (float)u02(rng);
		float phi = (float)u02(rng);

		boids[index].vel = (float)u01(rng)*glm::vec3(sin(theta)*cos(phi), sin(theta)*sin(phi), cos(theta));
    }
}

//Generate randomized starting velocities in the XY plane
__global__
void generateRandomVelArray(int time, int N, boid * boids, float scale)
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if(index < N)
    {
        glm::vec3 rand = (scale - BUFFER)*(generateRandomNumberFromThread(time, index) - 0.5f);
        boids[index].vel = rand;
    }
}

__device__ glm::vec3 calcAlign(int N, boid my_boid, boid* them_boids){
	glm::vec3 v = glm::vec3(0.0f);
	int numNeighbors = 0;

	int numBlocks = (int)ceil((float)N/blockSize);
	int index;
	float d, theta;

	__shared__ boid s[blockSize];
	for(int i = 0; i < numBlocks; i++){
		index = i * blockSize + threadIdx.x;
		if(index < N) s[threadIdx.x] = them_boids[index];
		__syncthreads();

		for(int j = 0; j < blockSize && j + i * blockSize < N; j++){
			d = glm::length(my_boid.pos - s[j].pos);
			theta = glm::dot(glm::normalize(-my_boid.vel), glm::normalize(s[j].pos - my_boid.pos));
			if(d < my_boid.r && (abs(theta) < abs(cos(BLIND_ANGLE)))){
				v += s[j].vel;
				++numNeighbors;
			}
		}
	}

	return 1.0f * ((1.0f / numNeighbors * v) - my_boid.vel);
}

__device__ void alignment(int N, boid* boids, glm::vec3* vel){
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	if(index < N){
		boid my_boid = boids[index];
		glm::vec3 v = calcAlign(N, my_boid, boids);
		vel[index] += v;
	}
}

__device__ glm::vec3 calcCohesion(int N, boid my_boid, boid* them_boids){
	glm::vec3 center = glm::vec3(0.0f);
	int numNeighbors = 0;
	int numBlocks = (int)ceil((float)N/blockSize);

	int index;
	float d, theta;

	__shared__ boid s[blockSize];

	for(int i = 0; i < numBlocks; i++){
		index = i * blockSize + threadIdx.x;

		if(index < N) s[threadIdx.x] = them_boids[index];
		__syncthreads();

		for(int j = 0; j < blockSize && i * blockSize + j < N; j++){
			d = glm::length(my_boid.pos - s[j].pos);
			theta = glm::dot(glm::normalize(-my_boid.vel), glm::normalize(s[j].pos - my_boid.pos));
			if(d < my_boid.r && (abs(theta) < abs(cos(BLIND_ANGLE)))){
				center += s[j].pos;
				++numNeighbors;
			}
		}
	}

	center = 1.0f / numNeighbors * center;
	return 1.0f * (center - my_boid.pos);
}

__device__ void cohesion(int N, boid* boids, glm::vec3* vel){
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	if(index < N){
		boid my_boid = boids[index];
		glm::vec3 v = calcCohesion(N, my_boid, boids);
		vel[index] += v;
	}
}

__device__ glm::vec3 calcSeparation(int N, boid my_boid, boid* them_boids){
	glm::vec3 v = glm::vec3(0.0f);
	int numNeighbors = 0;
	int numBlocks = (int)ceil((float)N/blockSize);
	int index;
	float d, theta;

	__shared__ boid s[blockSize];

	for(int i = 0; i < numBlocks; i++){
		index = i * blockSize + threadIdx.x;
		if(index < N) s[threadIdx.x] = them_boids[index];
		__syncthreads();

		for(int j = 0; j < blockSize && j + i * blockSize < N; j++){
			d = glm::length(my_boid.pos - s[j].pos);
			theta = glm::dot(glm::normalize(-my_boid.vel), glm::normalize(s[j].pos - my_boid.pos));
			if(d < my_boid.r && (abs(theta) < abs(cos(BLIND_ANGLE)))){
				v += s[j].pos - my_boid.pos;
				++numNeighbors;
			}
		}
	}
	v = 1.0f / numNeighbors * v;
	return -1.02f * v;
}

__device__ void separation(int N, boid* boids, glm::vec3* vel){
	int index= (blockIdx.x * blockDim.x) + threadIdx.x;
	if(index < N){
		boid my_boid = boids[index];
		glm::vec3 v = calcSeparation(N, my_boid, boids);
		vel[index] += v;
	}
}

__device__ glm::vec3 wander(int N, float dt, boid my_boid){
	int index = threadIdx.x + (blockDim.x * blockIdx.x);

	if(index < N){
		glm::vec3 new_center = my_boid.pos + my_boid.vel * dt;
		glm::vec3 displacement;

	    thrust::default_random_engine rng(hash(index * (timeStep + 1) * new_center.x));
		thrust::uniform_real_distribution<float> u01(-PI, PI);
		thrust::uniform_real_distribution<float> r(MIN_WANDER, MAX_WANDER);

		float d = (float) r(rng);

		// x-y axis circle
		displacement = glm::vec3(d * cos((float)u01(rng)), d * sin((float)u01(rng)), 0.0f);

		// x-z axis circle
		displacement += glm::vec3(d * cos((float)u01(rng)), 0.0f, d * sin((float)u01(rng)));
		//float theta = (float)u01(rng);
		//float phi = (float)u01(rng);

		//displacement = d * glm::vec3(sin(theta)*cos(phi), sin(theta)*sin(phi), cos(theta));

		displacement = glm::dot(displacement, my_boid.vel) < .001 ? -1.0f * displacement : displacement;

		return displacement;
	}
}

__device__ void avoidWalls(int N, float dt, boid my_boid, glm::vec3* acc){
	glm::vec3 a = glm::vec3(0.0f);
	int numBlocks = (int)ceil((float)N/blockSize);
	int index = threadIdx.x + (blockDim.x * blockIdx.x);
	float axis, d;
	glm::vec3 wall_intersect;
	thrust::default_random_engine rng(hash(N * index * my_boid.vel.y));
	thrust::uniform_real_distribution<float> u01(MIN_VELOCITY, MAX_VELOCITY);

	for(int i = 0; i < 3; i++){
		axis = my_boid.pos[i];

		if(abs(axis) > (WALL_DEPTH - BUFFER) && abs(axis) < WALL_DEPTH){
			glm::vec3 wall_pos = my_boid.pos;
			wall_pos[i] = axis < -(WALL_DEPTH - BUFFER) ? -WALL_DEPTH : WALL_DEPTH;
			acc[index] += my_boid.pos - wall_pos - my_boid.vel;
		}

		if(abs(axis) >= WALL_DEPTH){
			glm::vec3 wall_pos = my_boid.pos;
			wall_pos[i] = axis < -WALL_DEPTH ? -WALL_DEPTH : WALL_DEPTH;
			acc[index] += 2.0f * wall_pos - my_boid.pos;
		}
	}
}

//Simple Euler integration scheme
__global__
void updateF(int N, float dt, boid * boids, glm::vec3* acc)
{
    int index = threadIdx.x + (blockIdx.x * blockDim.x);
    boid my_pos;
    glm::vec3 accel;

    if(index < N){
		my_pos = boids[index];
		acc[index] = wander(N, dt, my_pos);
		avoidWalls(N, dt, my_pos, acc);
	}
	
	alignment(N, boids, acc);
	cohesion(N, boids, acc);
	separation(N, boids, acc);
}

__device__
void eulerIntegrate(float dt, boid* boids, glm::vec3* acc){
	int index = threadIdx.x + (blockIdx.x * blockDim.x);
	thrust::default_random_engine rng(hash(index * boids[index].vel.y));
	thrust::uniform_real_distribution<float> u01(MIN_VELOCITY, MAX_VELOCITY);

    boids[index].vel   = glm::clamp(boids[index].vel + (acc[index]   * dt), -MAX_VELOCITY, MAX_VELOCITY);
    boids[index].pos += boids[index].vel * dt;
}

__device__
void RK4Integrate(float dt, glm::vec4* pos, glm::vec3* acc){
}

__global__
void updateS(int N, float dt, boid * boids, glm::vec3 * acc)
{
    int index = threadIdx.x + (blockIdx.x * blockDim.x);
    if( index < N )
    {
		INTEGRATE(dt, boids, acc);
    }
}

//Update the vertex buffer object
//(The VBO is where OpenGL looks for the positions for the planets)
__global__
void sendToVBO(int N, boid * boids, float * vbo, float* vbo_n, int width, int height, float s_scale)
{
    int index = threadIdx.x + (blockIdx.x * blockDim.x);

    float c_scale_w = -2.0f / s_scale;
    float c_scale_h = -2.0f / s_scale;
	float c_scale_d = -2.0f / s_scale;

    if(index<N)
    {
		glm::vec3 pos = boids[index].pos;
		glm::vec3 vel = boids[index].vel;

        vbo[4*index+0] = pos.x*c_scale_w;
        vbo[4*index+1] = pos.y*c_scale_h;
        vbo[4*index+2] = pos.z*c_scale_d;
        vbo[4*index+3] = 1;

		vbo_n[4*index+0] = vel.x*c_scale_w;
        vbo_n[4*index+1] = vel.y*c_scale_h;
        vbo_n[4*index+2] = vel.z*c_scale_d;
        vbo_n[4*index+3] = 1;
    }
}

/*************************************
 * Wrappers for the __global__ calls *
 *************************************/

//Initialize memory, update some globals
void initCuda(int N)
{
    numObjects = N;
    dim3 fullBlocksPerGrid((int)ceil(float(N)/float(blockSize)));

    cudaMalloc((void**)&dev_boids, N*sizeof(boid));
    checkCUDAErrorWithLine("Kernel failed!");
    cudaMalloc((void**)&dev_acc, N*sizeof(glm::vec3));
    checkCUDAErrorWithLine("Kernel failed!");

    generateRandomPosArray<<<fullBlocksPerGrid, blockSize>>>(1, numObjects, dev_boids, scene_scale);
    checkCUDAErrorWithLine("Kernel failed!");
    generateCircularVelArray<<<fullBlocksPerGrid, blockSize>>>(2, numObjects, dev_boids);
    checkCUDAErrorWithLine("Kernel failed!");
    cudaThreadSynchronize();
}

void cudaNBodyUpdateWrapper(float dt)
{
    dim3 fullBlocksPerGrid((int)ceil(float(numObjects)/float(blockSize)));
    updateF<<<fullBlocksPerGrid, blockSize, blockSize*sizeof(glm::vec4)>>>(numObjects, dt, dev_boids, dev_acc);
    checkCUDAErrorWithLine("Kernel failed!");
    updateS<<<fullBlocksPerGrid, blockSize>>>(numObjects, dt, dev_boids, dev_acc);
    checkCUDAErrorWithLine("Kernel failed!");
    cudaThreadSynchronize();
}

void cudaUpdateVBO(float * vbodptr, float* vbodptrn, int width, int height)
{
    dim3 fullBlocksPerGrid((int)ceil(float(numObjects)/float(blockSize)));
    sendToVBO<<<fullBlocksPerGrid, blockSize>>>(numObjects, dev_boids, vbodptr, vbodptrn, width, height, scene_scale);
	timeStep++;
    cudaThreadSynchronize();
}


