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
const float planetMass = 3e9;
const __device__ float starMass = 6e10;

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
	//
	//                       ->
	//->			G*m_them*r   
    //a  = ---------------------------
    //         (||r||^2+eps^2)^3/2     
	glm::vec4 r = them-us;
    float r_mag_2 = r.x*r.x+r.y*r.y+r.z*r.z;
	float eps_2 = 0.1;
	float d = r_mag_2+eps_2;
	float a = them.w*G/sqrtf(d*d*d);
    return a*glm::vec3(r);//use diff as direction;;
}

//TODO: Core force calc kernel global memory
__device__ 
glm::vec3 naiveAcc(int N, glm::vec4 my_pos, glm::vec4 * their_pos)
{
    glm::vec3 acc = calculateAcceleration(my_pos, glm::vec4(0,0,0,starMass));
	

	for(int i = 0; i < N; ++i)
	{
		acc = acc + calculateAcceleration(my_pos, their_pos[i]);
	}
    return acc;
}



//TODO: Core force calc kernel shared memory
__device__ 
glm::vec3 sharedMemAcc(int N, glm::vec4 my_pos, glm::vec4 * their_pos)
{
	extern __shared__ glm::vec4 shPosition[];  
	
	//Do initial hardcoded calcuation from star position
	glm::vec3 acc = calculateAcceleration(my_pos, glm::vec4(0,0,0,starMass));

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
			shPosition[threadIdx.x] = their_pos[loadIndex];

		//Wait for load to finish
		__syncthreads();

		//Perform update for entire tile using shared memory
		//No bank conflicts because this is broadcast
		for(int i = 0; i < blockDim.x; ++i)
		{
			int idx = tileOffset+i;
			if(idx < N)
				acc = acc + calculateAcceleration(my_pos, shPosition[idx]);
			else
				break;
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
    glm::vec4 my_pos;
    glm::vec3 accel;

    if(index < N) my_pos = pos[index];

    accel = ACC(N, my_pos, pos);

    if(index < N) acc[index] = accel;
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
        //Position
		vbo[boidVBO_PositionOffset + boidVBOStride*index + 0] =  pos[index].x*c_scale_w;
		vbo[boidVBO_PositionOffset + boidVBOStride*index + 1] =  pos[index].y*c_scale_h;
		vbo[boidVBO_PositionOffset + boidVBOStride*index + 2] = 0.0f;
		vbo[boidVBO_PositionOffset + boidVBOStride*index + 3] = 1.0f;

		//Up
		vbo[boidVBO_UpOffset + boidVBOStride*index + 0] = 0.0f;
		vbo[boidVBO_UpOffset + boidVBOStride*index + 1] = 0.0f;
		vbo[boidVBO_UpOffset + boidVBOStride*index + 2] = 1.0f;

		//Forward
		vbo[boidVBO_ForwardOffset + boidVBOStride*index + 0] = 1.0f;
		vbo[boidVBO_ForwardOffset + boidVBOStride*index + 1] = 0.0f;
		vbo[boidVBO_ForwardOffset + boidVBOStride*index + 2] = 0.0f;

		//Color
		vbo[boidVBO_ColorOffset + boidVBOStride*index + 0] = 1.0f;
		vbo[boidVBO_ColorOffset + boidVBOStride*index + 1] = 1.0f;
		vbo[boidVBO_ColorOffset + boidVBOStride*index + 2] = 0.0f;

		//Shape
		vbo[boidVBO_ShapeOffset + boidVBOStride*index + 0] = 1.0f;//Length
		vbo[boidVBO_ShapeOffset + boidVBOStride*index + 1] = 1.0f;//Wingspan
		vbo[boidVBO_ShapeOffset + boidVBOStride*index + 2] = 0.0f;//Delta
		vbo[boidVBO_ShapeOffset + boidVBOStride*index + 3] = 0.0f;//Wing Deflection
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

    generateRandomPosArray<<<fullBlocksPerGrid, blockSize>>>(1, numObjects, dev_pos, scene_scale, planetMass);
    checkCUDAErrorWithLine("Kernel failed!");
    generateCircularVelArray<<<fullBlocksPerGrid, blockSize>>>(2, numObjects, dev_vel, dev_pos);
    checkCUDAErrorWithLine("Kernel failed!");
    cudaThreadSynchronize();
}

void cudaNBodyUpdateWrapper(float dt)
{
    dim3 fullBlocksPerGrid((int)ceil(float(numObjects)/float(blockSize)));
    updateF<<<fullBlocksPerGrid, blockSize, blockSize*sizeof(glm::vec4)>>>(numObjects, dt, dev_pos, dev_vel, dev_acc);
    checkCUDAErrorWithLine("Kernel Update failed!");
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


