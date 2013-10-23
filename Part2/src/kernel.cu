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
dim3 threadsPerBlock(BLOCK_SIZE);

int numObjects;
const float planetMass = 3e8;
const __device__ float starMass = 5e10;

const float scene_scale = 2e2; //size of the height map in simulation space

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
void generateRandomPosArray(int time, int N, glm::vec4 * arr, float scale, float mass)
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
__global__
void generateCircularVelArray(int time, int N, glm::vec3 * arr, glm::vec4 * pos)
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

__device__
glm::vec3 calculateRandomDirection(float xi) {
    return glm::vec3(cos(TWO_PI*xi), sin(TWO_PI*xi), 0.0f);
}

__global__
void generateInitialVelArray(int time, int N, glm::vec3 * arr)
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if(index < N)
    {
        glm::vec3 rand = generateRandomNumberFromThread(time, index);
        arr[index] = calculateRandomDirection(rand.z) * (float)STARTING_VEL;
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

//TODO: Determine force between two bodies
__device__
glm::vec3 calculateAcceleration(glm::vec4 us, glm::vec4 them)
{
    //    G*m_us*m_them
    //F = -------------
    //         r^2
    //
    //    G*m_us*m_them   G*m_them
    //a = ------------- = --------
    //      m_us*r^2        r^2

	glm::vec3 p_us = glm::vec3(us.x, us.y, us.z);
	glm::vec3 p_them = glm::vec3(them.x, them.y, them.z);

	float m_them = them.w;
	float r = glm::distance(p_us, p_them);
	return glm::normalize(p_them - p_us) * (float)G*m_them/(r*r);
}

// FLOCKING

// Cohesion

__device__
glm::vec3 cohesion(int N, glm::vec3 my_pos, glm::vec4 * all_pos) {\
    glm::vec3 other_pos;
    glm::vec3 mean_pos = glm::vec3(0.0f, 0.0f, 0.0f);
    int N_flock = 0;
    int i;
	for (i=0; i<N; ++i) {
        other_pos = glm::vec3(all_pos[i].x, all_pos[i].y, all_pos[i].z);
        if (glm::distance(my_pos, other_pos) < FLOCKING_RADIUS) {
            ++N_flock;
	        mean_pos += other_pos;
        }
	}
	mean_pos = mean_pos/(float)N_flock;
    return (mean_pos - my_pos) * (float)COHESION_COEFF;
}

__device__
glm::vec3 alignment(int N, glm::vec3 my_pos, glm::vec3 my_vel, glm::vec4 * all_pos, glm::vec3 * all_vel) {
    glm::vec3 other_pos;
    glm::vec3 mean_vel = glm::vec3(0.0f, 0.0f, 0.0f);
    int N_flock = 0;
  	int i;
	for (i=0; i<N; ++i) {
        other_pos = glm::vec3(all_pos[i].x, all_pos[i].y, all_pos[i].z);
        if (glm::distance(my_pos, other_pos) < FLOCKING_RADIUS) {
            ++N_flock;
	        mean_vel += all_vel[i];
        }
	}
	mean_vel = mean_vel/(float)N_flock*(float)ALIGNMENT_BONUS;
    return (mean_vel - my_vel) * (float)ALIGNMENT_COEFF;
}

__device__
glm::vec3 separation(int N, glm::vec3 my_pos, glm::vec4 * all_pos) {
    glm::vec3 other_pos;
    glm::vec3 separation_vec = glm::vec3(0.0f, 0.0f, 0.0f);
	int i;
	for (i=0; i<N; ++i) {
        other_pos = glm::vec3(all_pos[i].x, all_pos[i].y, all_pos[i].z);
        float sep = glm::distance(my_pos, other_pos);
        if (sep < (float)SEPARATION_RADIUS && sep > (float)SEPARATION_EPSILON) {
	        separation_vec += glm::normalize(my_pos - other_pos)/(sep*sep)*(float)SEPARATION_COEFF;
        }
	}
    return separation_vec;
}

//TODO: Core force calc kernel global memory

__device__ 
glm::vec3 naiveAcc(int N, glm::vec4 my_pos, glm::vec4 * their_pos)
{
    glm::vec3 acc = calculateAcceleration(my_pos, glm::vec4(0,0,0,starMass));
	int i;
	for (i=0; i<N; ++i) {
		if (glm::distance(my_pos, their_pos[i]) > 10.0) {
			acc += calculateAcceleration(my_pos, their_pos[i]);
		}
	}
    return acc;
}

//TODO: Core force calc kernel shared memory
__device__ 
glm::vec3 sharedMemAcc(int N, glm::vec4 my_pos, glm::vec4 * their_pos)
{
	int thread_index = threadIdx.x;
	extern __shared__ glm::vec4 shared_pos[];
	int num_splits = ceil((float)N/(float)(BLOCK_SIZE));

    glm::vec3 acc = calculateAcceleration(my_pos, glm::vec4(0,0,0,starMass));

	int i, j, pos_index = 0;
	for (i=0; i<num_splits; ++i) {
		if (i*BLOCK_SIZE + thread_index < N) {
			shared_pos[thread_index] = their_pos[i*BLOCK_SIZE + thread_index];
		}

		__syncthreads();

		for (j=0; j<BLOCK_SIZE; ++j) {
			++pos_index;
			if (pos_index >= N) {
				break;
			}
			if (glm::distance(my_pos, shared_pos[pos_index]) > 10.0) {
				acc += calculateAcceleration(my_pos, shared_pos[pos_index]);
			}
		}

		__syncthreads();
	}
    return acc;
}

//Simple Euler integration scheme
__global__
void updateF(int N, float dt, glm::vec4 * pos, glm::vec3 * vel, glm::vec3 * acc)
{
    int index = threadIdx.x + (blockIdx.x * blockDim.x);
    glm::vec3 my_pos;
    glm::vec3 my_vel;
    glm::vec3 accel;

    if(index < N) {
        my_pos = glm::vec3(pos[index].x, pos[index].y, pos[index].z);
        my_vel = vel[index];

        accel = cohesion(N, my_pos, pos);
        accel += alignment(N, my_pos, my_vel, pos, vel);
        accel += separation(N, my_pos, pos);
        if (glm::length(my_pos) > ATTRACTOR_RADIUS) {
            glm::vec3 attraction = -my_pos;
            glm::vec3 deflection = glm::cross(attraction, glm::vec3(0.0f, 0.0f, 1.0f));
            if (glm::dot(deflection, my_vel) > 0) {
                deflection = -deflection;
            }
            accel += (attraction*(1.0f-(float)ATTRACTOR_DEFLECTION) + deflection*(float)ATTRACTOR_DEFLECTION) * (float)ATTRACTOR_COEFF;
        }
        acc[index] = accel;
    }
}

__global__
void updateS(int N, float dt, glm::vec4 * pos, glm::vec3 * vel, glm::vec3 * acc)
{
    int index = threadIdx.x + (blockIdx.x * blockDim.x);
    if( index < N )
    {
        vel[index]   += acc[index]   * dt;
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
    glm::vec3 acc = ACC(N, glm::vec4((x-w2)/c_scale_w,(y-h2)/c_scale_h,0,1), pos);

    if(x<width && y<height)
    {
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
    dim3 fullBlocksPerGrid((int)ceil(float(N)/float(BLOCK_SIZE)));

    cudaMalloc((void**)&dev_pos, N*sizeof(glm::vec4));
    checkCUDAErrorWithLine("Kernel failed!");
    cudaMalloc((void**)&dev_vel, N*sizeof(glm::vec3));
    checkCUDAErrorWithLine("Kernel failed!");
    cudaMalloc((void**)&dev_acc, N*sizeof(glm::vec3));
    checkCUDAErrorWithLine("Kernel failed!");

    generateRandomPosArray<<<fullBlocksPerGrid, BLOCK_SIZE>>>(1, numObjects, dev_pos, scene_scale, planetMass);
    checkCUDAErrorWithLine("Kernel failed!");
    generateInitialVelArray<<<fullBlocksPerGrid, BLOCK_SIZE>>>(2, numObjects, dev_vel);
    checkCUDAErrorWithLine("Kernel failed!");
    cudaThreadSynchronize();
}

void cudaNBodyUpdateWrapper(float dt)
{
    dim3 fullBlocksPerGrid((int)ceil(float(numObjects)/float(BLOCK_SIZE)));
	printf("fullBlocksPerGrid: %d\n", fullBlocksPerGrid);
	printf("BLOCK_SIZE: %d\n", BLOCK_SIZE);
	printf("BLOCK_SIZE*sizeof(glm::vec4): %d\n", BLOCK_SIZE*sizeof(glm::vec4));
    updateF<<<fullBlocksPerGrid, BLOCK_SIZE, BLOCK_SIZE*sizeof(glm::vec4)>>>(numObjects, dt, dev_pos, dev_vel, dev_acc);
    checkCUDAErrorWithLine("Kernel failed!");
    updateS<<<fullBlocksPerGrid, BLOCK_SIZE>>>(numObjects, dt, dev_pos, dev_vel, dev_acc);
    checkCUDAErrorWithLine("Kernel failed!");
    cudaThreadSynchronize();
}

void cudaUpdateVBO(float * vbodptr, int width, int height)
{
    dim3 fullBlocksPerGrid((int)ceil(float(numObjects)/float(BLOCK_SIZE)));
    sendToVBO<<<fullBlocksPerGrid, BLOCK_SIZE>>>(numObjects, dev_pos, vbodptr, width, height, scene_scale);
    cudaThreadSynchronize();
}

void cudaUpdatePBO(float4 * pbodptr, int width, int height)
{
    dim3 fullBlocksPerGrid((int)ceil(float(width*height)/float(BLOCK_SIZE)));
    sendToPBO<<<fullBlocksPerGrid, BLOCK_SIZE, BLOCK_SIZE*sizeof(glm::vec4)>>>(numObjects, dev_pos, pbodptr, width, height, scene_scale);
    cudaThreadSynchronize();
}


