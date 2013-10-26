#include <stdio.h>
#include <cuda.h>
#include <cmath>
#include "glm/glm.hpp"
#include "utilities.h"
#include "kernel.h"

using namespace glm;

/*
#if SHARED == 1
    #define ACC(x,y,z) sharedMemAcc(x,y,z)
#else
    #define ACC(x,y,z) naiveAcc(x,y,z)
#endif
	*/

//GLOBALS
dim3 threadsPerBlock(blockSize);

int numObjects;
const float radius = 50.0f;

const float scene_scale = 500; //size of the height map in simulation space

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
void generateRandomPosArray(int time, int N, glm::vec4 * arr, float scale )
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if(index < N)
    {
        glm::vec3 rand = (scale-50)*(generateRandomNumberFromThread(time, index)-0.5f);
        arr[index].x = rand.x;
		arr[index].y = rand.y;
		arr[index].z = rand.z;
		arr[index].w = 0.0f;
    }
}

//Determine velocity from the distance from the center star. Not super physically accurate because 
//the mass ratio is too close, but it makes for an interesting looking scene
__global__
void generateCircularVelArray(int time, int N, glm::vec3 * arr, glm::vec4 * pos)
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if(index < N)
    {
       thrust::default_random_engine rng(hash((time + index) * N * index));
       thrust::uniform_real_distribution<float> u01(.1, 10);
       thrust::uniform_real_distribution<float> u02(-PI, PI);

		float theta = (float)u02(rng);
		float phi = (float)u02(rng);

        arr[index] = (float)u01(rng)*glm::vec3(sin(theta)*cos(phi), sin(theta)*sin(phi), cos(theta));
		/*
		glm::vec3 rand = 10.0f*(generateRandomNumberFromThread(time+index*N+threadIdx.x, index));
        arr[index].x = rand.x;
		arr[index].y = rand.y;
		arr[index].z = rand.z;*/
    }
}

//Generate randomized starting velocities in the XY plane
__global__
void generateRandomVelArray(int time, int N, glm::vec3 * arr, float scale)
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

__device__
vec3 calculateAlignment(int N, vec4 current_boid_pos, vec4* other_boids_pos, vec3 current_boid_vel, vec3* other_boids_vel, vec3* acc, float radius){
	int numInRadius = 0;
	int index;
	float distance;
	float angle;
	int numberOfBlocks = (int)ceil((float)N/blockSize);
	vec3 averageVelocity = vec3(0,0,0);

	__shared__ vec4 positions[blockSize];
	__shared__ vec3 velocities[blockSize];

	for(int i = 0; i < numberOfBlocks; i++){
		index = (i*blockSize)+threadIdx.x;
		if(index < N){
			positions[threadIdx.x] = other_boids_pos[index];
			velocities[threadIdx.x] = other_boids_vel[index];
		}
		__syncthreads();

		for(int j = 0; j < blockSize && j + i*blockSize < N; j++){
			distance = length(current_boid_pos - positions[j]);
			vec3 vectorDist = vec3(positions[j]-current_boid_pos);
			vectorDist = (-1.0f/distance)*vectorDist;
			vec3 normalCurrentBoidVel = (-1.0f/length(current_boid_vel))*current_boid_vel;
			angle = glm::dot(normalCurrentBoidVel,vectorDist);
			if(distance <= radius && abs(angle) < abs(cos((float)fieldOfView))){
				numInRadius++;
				averageVelocity += velocities[j];
			}
		}
	}

	if (numInRadius > 0)
		return 1.0f*(((1.0f/(float)numInRadius) * averageVelocity) - current_boid_vel);	//returns average velocity of birds within radius
	else
		return vec3(0,0,0);

}

__device__
void alignment(int N, vec4 * pos, vec3* vel, vec3* acc, float radius){
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;
	if (index < N){
		vec4 current_boid_pos = pos[index];
		vec3 current_boid_vel = vel[index];
		vec3 accel = calculateAlignment(N,current_boid_pos,pos,current_boid_vel,vel,acc,radius);
		acc[index] += accel;
		//printf("%f %f %f\n", acc[index][0], acc[index][1], acc[index][2]);
	}
}

__device__
vec3 calculateCohesion(int N, vec4 current_boid_pos, vec4* other_boids_pos, vec3 current_boid_vel, vec3* other_boids_vel, vec3* acc, float radius){
	int numInRadius = 0;
	int index;
	float distance;
	float angle;
	int numberOfBlocks = (int)ceil((float)N/blockSize);
	vec3 averagePosition = vec3(0,0,0);

	__shared__ vec4 positions[blockSize];
	__shared__ vec3 velocities[blockSize];

	for(int i = 0; i < numberOfBlocks; i++){
		index = (i*blockSize)+threadIdx.x;
		if(index < N){
			positions[threadIdx.x] = other_boids_pos[index];
			velocities[threadIdx.x] = other_boids_vel[index];
		}
		__syncthreads();

		for(int j = 0; j < blockSize && j + i*blockSize < N; j++){
			distance = length(current_boid_pos - positions[j]);
			vec3 vectorDist = vec3(positions[j]-current_boid_pos);
			vectorDist = (-1.0f/distance)*vectorDist;
			vec3 normalCurrentBoidVel = (-1.0f/length(current_boid_vel))*current_boid_vel;
			angle = glm::dot(normalCurrentBoidVel,vectorDist);
			if(distance <= radius && abs(angle) < abs(cos((float)fieldOfView))){
				numInRadius++;
				averagePosition += vec3(positions[j]);
			}
		}
	}

	if (numInRadius > 0)
		return 1.0f*(((1.0f/(float)numInRadius) * averagePosition) - vec3(current_boid_pos));	//returns average velocity of birds within radius
	else
		return vec3(0,0,0);
}

__device__
void cohesion(int N, vec4 * pos, vec3* vel, vec3* acc, float radius){
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;
	if (index < N){
		vec4 current_boid_pos = pos[index];
		vec3 current_boid_vel = vel[index];
		vec3 accel = calculateCohesion(N,current_boid_pos,pos,current_boid_vel,vel,acc,radius);
		acc[index] += accel;
		//printf("%f %f %f\n", acc[index][0], acc[index][1], acc[index][2]);
	}
}

__device__
vec3 calculateSeparation(int N, vec4 current_boid_pos, vec4* other_boids_pos, vec3 current_boid_vel, vec3* other_boids_vel, vec3* acc, float radius){
	int numInRadius = 0;
	int index;
	float distance;
	float angle;
	int numberOfBlocks = (int)ceil((float)N/blockSize);
	vec3 averageDirection = vec3(0,0,0);

	__shared__ vec4 positions[blockSize];
	__shared__ vec3 velocities[blockSize];

	for(int i = 0; i < numberOfBlocks; i++){
		index = (i*blockSize)+threadIdx.x;
		if(index < N){
			positions[threadIdx.x] = other_boids_pos[index];
			velocities[threadIdx.x] = other_boids_vel[index];
		}
		__syncthreads();

		for(int j = 0; j < blockSize && j + i*blockSize < N; j++){
			distance = length(current_boid_pos - positions[j]);
			vec3 vectorDist = vec3(positions[j]-current_boid_pos);
			vectorDist = (-1.0f/distance)*vectorDist;
			vec3 normalCurrentBoidVel = (-1.0f/length(current_boid_vel))*current_boid_vel;
			angle = glm::dot(normalCurrentBoidVel,vectorDist);
			if(distance <= radius && abs(angle) < abs(cos((float)fieldOfView))){
				numInRadius++;
				averageDirection += vec3(current_boid_pos-positions[j]);
			}
		}
	}

	if (numInRadius > 0)
		return (1.0f/(float)numInRadius) * averageDirection;	//returns average velocity of birds within radius
	else
		return vec3(0,0,0);
}


__device__
void separation(int N, vec4 * pos, vec3* vel, vec3* acc, float radius){
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;
	if (index < N){
		vec4 current_boid_pos = pos[index];
		vec3 current_boid_vel = vel[index];
		vec3 avgDir = calculateSeparation(N,current_boid_pos,pos,current_boid_vel,vel,acc,radius);
		acc[index] += 1.0f*(avgDir);
		//printf("%f %f %f\n", acc[index][0], acc[index][1], acc[index][2]);
	}
}

//Simple Euler integration scheme
__global__
void updateF(int time, int N, float dt, glm::vec4 * pos, glm::vec3 * vel, glm::vec3 * acc,float radius)
{
    int index = threadIdx.x + (blockIdx.x * blockDim.x);
    glm::vec4 my_pos;
	glm::vec3 my_vel;
	glm::vec3 accel;

    if(index < N) {
		my_pos = pos[index];
		my_vel = vel[index];

		acc[index] =(generateRandomNumberFromThread(time, index) - 0.5f)*my_vel;
		if(length(my_pos) > 500){
			vel[index] = reflect(my_vel,normalize(vec3(-my_pos)));
			pos[index] = normalize(pos[index])*499;
		}
		//acc[index] = vec3(0,0,0);//2.0f*(generateRandomNumberFromThread(time, index));
	}
	 
	alignment(N,pos,vel,acc,radius);
	cohesion(N,pos,vel,acc,radius);
	separation(N,pos,vel,acc,radius);
}

__global__
void updateS(int N, float dt, glm::vec4 * pos, glm::vec3 * vel, glm::vec3 * acc)
{
    int index = threadIdx.x + (blockIdx.x * blockDim.x);
    if( index < N)
    {
        vel[index]   += acc[index]   * dt;
		vel[index] = clamp(vel[index], -6, 6);
        pos[index].x += vel[index].x * dt;
        pos[index].y += vel[index].y * dt;
        pos[index].z += vel[index].z * dt;
    }
}

//Update the vertex buffer object
//(The VBO is where OpenGL looks for the positions for the planets)
__global__
void sendToVBO(int N, glm::vec4 * pos, float * vbo, int width, int height, float s_scale)
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
__global__
void sendToPBO(int N, glm::vec4 * pos, float4 * pbo, int width, int height, float s_scale)
{
    int index = threadIdx.x + (blockIdx.x * blockDim.x);
    int x = index % width;
    int y = index / width;
    float w2 = width / 2.0;
    float h2 = height / 2.0;

    float c_scale_w = width / s_scale;
    float c_scale_h = height / s_scale;

    glm::vec3 color(0.05, 0.15, 0.3);
    ///glm::vec3 acc = ACC(N, glm::vec4((x-w2)/c_scale_w,(y-h2)/c_scale_h,0,1), pos);

    if(x<width && y<height)
    {
        //float mag = sqrt(sqrt(acc.x*acc.x + acc.y*acc.y + acc.z*acc.z));
		float mag = 1.0f;
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
    dim3 fullBlocksPerGrid((int)ceil(float(N)/float(blockSize)));

    cudaMalloc((void**)&dev_pos, N*sizeof(glm::vec4));
    checkCUDAErrorWithLine("Kernel failed!");
    cudaMalloc((void**)&dev_vel, N*sizeof(glm::vec3));
    checkCUDAErrorWithLine("Kernel failed!");
    cudaMalloc((void**)&dev_acc, N*sizeof(glm::vec3));
    checkCUDAErrorWithLine("Kernel failed!");

    generateRandomPosArray<<<fullBlocksPerGrid, blockSize>>>(1, numObjects, dev_pos, scene_scale);
    checkCUDAErrorWithLine("Kernel failed!");
    generateCircularVelArray<<<fullBlocksPerGrid, blockSize>>>(2, numObjects, dev_vel, dev_pos);
    checkCUDAErrorWithLine("Kernel failed!");
    cudaThreadSynchronize();
}

void cudaNBodyUpdateWrapper(float dt)
{
    dim3 fullBlocksPerGrid((int)ceil(float(numObjects)/float(blockSize)));
    updateF<<<fullBlocksPerGrid, blockSize, blockSize*sizeof(glm::vec4)>>>(3, numObjects, dt, dev_pos, dev_vel, dev_acc, radius);
    checkCUDAErrorWithLine("Kernel failed!");
    updateS<<<fullBlocksPerGrid, blockSize>>>(numObjects, dt, dev_pos, dev_vel, dev_acc);
    checkCUDAErrorWithLine("Kernel failed!");
    cudaThreadSynchronize();
}

void cudaUpdateVBO(float * vbodptr, int width, int height)
{
    dim3 fullBlocksPerGrid((int)ceil(float(numObjects)/float(blockSize)));
    sendToVBO<<<fullBlocksPerGrid, blockSize>>>(numObjects, dev_pos, vbodptr, width, height, scene_scale);
    cudaThreadSynchronize();
}

void cudaUpdatePBO(float4 * pbodptr, int width, int height)
{
    dim3 fullBlocksPerGrid((int)ceil(float(width*height)/float(blockSize)));
    sendToPBO<<<fullBlocksPerGrid, blockSize, blockSize*sizeof(glm::vec4)>>>(numObjects, dev_pos, pbodptr, width, height, scene_scale);
    cudaThreadSynchronize();
}


