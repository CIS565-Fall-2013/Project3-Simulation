#include <stdio.h>
#include <cuda.h>
#include <cmath>
#include "glm/glm.hpp"
#include "utilities.h"
#include "kernel.h"

#if SHARED == 1
    #define ACC(a,b,c,d) sharedMemAcc(a,b,c,d)
#else
    #define ACC(a,b,c,d) naiveAcc(a,b,c,d)
#endif

#define NUM_NEIGHBORS 12
#define POINT_MASS 10
#define SCENE_SCALE 1e2
#define SPHERE_RADIUS 25

//GLOBALS
dim3 threadsPerBlock(blockSize);

int numParticles;
int numSprings;

glm::vec3 * dev_pos;
glm::vec3 * dev_vel;
glm::vec3 * dev_acc;
glm::vec3 * dev_spr;

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

// get 1-D array index from 2-D (X,Y) index
__host__ __device__
int getIndex(int x, int y, int width)
{
	return y*width+x;
}

//Function that generates static.
__host__ __device__ 
glm::vec3 generateRandomNumberFromThread(float time, int index)
{
    thrust::default_random_engine rng(hash(index*time));
    thrust::uniform_real_distribution<float> u01(0,1);

    return glm::vec3((float) u01(rng), (float) u01(rng), (float) u01(rng));
}

//Generate uniform grid of starting positions in XY plane for particles
__global__
void generateClothPosArray(int N, glm::vec3 * pos_arr)
{
	int clothWidth = floor(sqrt((double)N));
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if(index < N)
    {
		// get x- and y-index of particle in 2D grid
		int x_idx = index % clothWidth;
		int y_idx = index / clothWidth;

		// get x- and y-position in real space
		float x_pos = (x_idx + 0.5) * (SCENE_SCALE / clothWidth) - (SCENE_SCALE/2.0f);
		float y_pos = (y_idx + 0.5) * (SCENE_SCALE / clothWidth) - (SCENE_SCALE/2.0f);

        pos_arr[index].x = x_pos;
        pos_arr[index].y = y_pos;
        pos_arr[index].z = -30.0f;
    }
}

// Generate springs between each particle and its 12 neighbors
__global__
void generateSpringArray(int N, glm::vec3 * spring_arr, glm::vec3 * pos_arr, 
						 float k_structure, float k_shear, float k_bend)
{
	/*---------------------------------------*/
	/* Spring Object Vector Format:			 */
	/*	spring.x = opposite particle index	 */
	/*	spring.y = spring constant "k"		 */
	/*	spring.z = spring rest length "l0"	 */
	/*---------------------------------------*/

	int clothWidth = floor(sqrt(double(N)));

    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if(index < N)
    {
		// own position
		glm::vec3 self = pos_arr[index];

		// get x- and y-index of particle in 2D grid
		int X = index % clothWidth;
		int Y = index / clothWidth;

		/*--------------------------------*/
		/* create four structure springs  */
		/*--------------------------------*/

		// get indices of structure neighbors
		int idxStructNbr1 = getIndex(X-1,Y,clothWidth);	if (X-1 < 0)		   { idxStructNbr1 = -1; }
		int idxStructNbr2 = getIndex(X+1,Y,clothWidth);	if (X+1 >= clothWidth) { idxStructNbr2 = -1; }
		int idxStructNbr3 = getIndex(X,Y-1,clothWidth); if (Y-1 < 0)		   { idxStructNbr3 = -1; }
		int idxStructNbr4 = getIndex(X,Y+1,clothWidth); if (Y+1 >= clothWidth) { idxStructNbr4 = -1; }
		
		// get distances to structure neighbors
		float distToStructNbr1, distToStructNbr2, distToStructNbr3, distToStructNbr4;
		if (idxStructNbr1 < 0) { distToStructNbr1 = -1.0f; }
		else { glm::vec3 diff = pos_arr[idxStructNbr1] - self; distToStructNbr1 = sqrt(glm::dot(diff, diff)); }
		if (idxStructNbr2 < 0) { distToStructNbr2 = -1.0f; }
		else { glm::vec3 diff = pos_arr[idxStructNbr2] - self; distToStructNbr2 = sqrt(glm::dot(diff, diff)); }
		if (idxStructNbr3 < 0) { distToStructNbr3 = -1.0f; }
		else { glm::vec3 diff = pos_arr[idxStructNbr3] - self; distToStructNbr3 = sqrt(glm::dot(diff, diff)); }
		if (idxStructNbr4 < 0) { distToStructNbr4 = -1.0f; }
		else { glm::vec3 diff = pos_arr[idxStructNbr4] - self; distToStructNbr4 = sqrt(glm::dot(diff, diff)); }

		// form structure spring vectors
		glm::vec3 structure_spring_1(idxStructNbr1, k_structure, distToStructNbr1);
		glm::vec3 structure_spring_2(idxStructNbr2, k_structure, distToStructNbr2);
		glm::vec3 structure_spring_3(idxStructNbr3, k_structure, distToStructNbr3);
		glm::vec3 structure_spring_4(idxStructNbr4, k_structure, distToStructNbr4);

		// store structure spring vectors
		spring_arr[index*NUM_NEIGHBORS + 0] = structure_spring_1;
		spring_arr[index*NUM_NEIGHBORS + 1] = structure_spring_2;
		spring_arr[index*NUM_NEIGHBORS + 2] = structure_spring_3;
		spring_arr[index*NUM_NEIGHBORS + 3] = structure_spring_4;
		

		/*--------------------------------*/
		/* create four shear springs      */
		/*--------------------------------*/
		// get indices of structure neighbors
		int idxShearNbr1 = getIndex(X-1,Y-1,clothWidth); if (X-1 < 0 || Y-1 < 0)					{ idxShearNbr1 = -1; }
		int idxShearNbr2 = getIndex(X+1,Y-1,clothWidth); if (X+1 >= clothWidth || Y-1 < 0)			{ idxShearNbr2 = -1; }
		int idxShearNbr3 = getIndex(X-1,Y+1,clothWidth); if (X-1 < 0 || Y+1 >= clothWidth)			{ idxShearNbr3 = -1; }
		int idxShearNbr4 = getIndex(X-1,Y+1,clothWidth); if (X+1 >=clothWidth || Y+1 >= clothWidth) { idxShearNbr4 = -1; }
		
		// get distances to structure neighbors
		float distToShearNbr1, distToShearNbr2, distToShearNbr3, distToShearNbr4;
		if (idxShearNbr1 < 0) { distToShearNbr1 = -1.0f; }
		else { glm::vec3 diff = pos_arr[idxShearNbr1] - self; distToShearNbr1 = sqrt(glm::dot(diff, diff)); }
		if (idxShearNbr2 < 0) { distToShearNbr2 = -1.0f; }
		else { glm::vec3 diff = pos_arr[idxShearNbr2] - self; distToShearNbr2 = sqrt(glm::dot(diff, diff)); }
		if (idxShearNbr3 < 0) { distToShearNbr3 = -1.0f; }
		else { glm::vec3 diff = pos_arr[idxShearNbr3] - self; distToShearNbr3 = sqrt(glm::dot(diff, diff)); }
		if (idxShearNbr4 < 0) { distToShearNbr4 = -1.0f; }
		else { glm::vec3 diff = pos_arr[idxShearNbr4] - self; distToShearNbr4 = sqrt(glm::dot(diff, diff)); }

		// form structure spring vectors
		glm::vec3 shear_spring_1(idxShearNbr1, k_shear, distToShearNbr1);
		glm::vec3 shear_spring_2(idxShearNbr2, k_shear, distToShearNbr2);
		glm::vec3 shear_spring_3(idxShearNbr3, k_shear, distToShearNbr3);
		glm::vec3 shear_spring_4(idxShearNbr4, k_shear, distToShearNbr4);

		// store structure spring vectors
		spring_arr[index*NUM_NEIGHBORS + 4] = shear_spring_1;
		spring_arr[index*NUM_NEIGHBORS + 5] = shear_spring_2;
		spring_arr[index*NUM_NEIGHBORS + 6] = shear_spring_3;
		spring_arr[index*NUM_NEIGHBORS + 7] = shear_spring_4;
		

		/*--------------------------------*/
		/* create four bend springs       */
		/*--------------------------------*/
		// get indices of structure neighbors
		int idxBendNbr1 = getIndex(X-2,Y,clothWidth); if (X-2 < 0)		     { idxBendNbr1 = -1; }
		int idxBendNbr2 = getIndex(X+2,Y,clothWidth); if (X+2 >= clothWidth) { idxBendNbr2 = -1; }
		int idxBendNbr3 = getIndex(X,Y-2,clothWidth); if (Y-2 < 0)		     { idxBendNbr3 = -1; }
		int idxBendNbr4 = getIndex(X,Y+2,clothWidth); if (Y+2 >= clothWidth) { idxBendNbr4 = -1; }
		
		// get distances to structure neighbors
		float distToBendNbr1, distToBendNbr2, distToBendNbr3, distToBendNbr4;
		if (idxBendNbr1 < 0) { distToBendNbr1 = -1.0f; }
		else { glm::vec3 diff = pos_arr[idxBendNbr1] - self; distToBendNbr1 = sqrt(glm::dot(diff, diff)); }
		if (idxBendNbr2 < 0) { distToBendNbr2 = -1.0f; }
		else { glm::vec3 diff = pos_arr[idxBendNbr2] - self; distToBendNbr2 = sqrt(glm::dot(diff, diff)); }
		if (idxBendNbr3 < 0) { distToBendNbr3 = -1.0f; }
		else { glm::vec3 diff = pos_arr[idxBendNbr3] - self; distToBendNbr3 = sqrt(glm::dot(diff, diff)); }
		if (idxBendNbr4 < 0) { distToBendNbr4 = -1.0f; }
		else { glm::vec3 diff = pos_arr[idxBendNbr4] - self; distToBendNbr4 = sqrt(glm::dot(diff, diff)); }

		// form structure spring vectors
		glm::vec3 bend_spring_1(idxBendNbr1, k_bend, distToBendNbr1);
		glm::vec3 bend_spring_2(idxBendNbr2, k_bend, distToBendNbr2);
		glm::vec3 bend_spring_3(idxBendNbr3, k_bend, distToBendNbr3);
		glm::vec3 bend_spring_4(idxBendNbr4, k_bend, distToBendNbr4);

		// store structure spring vectors
		spring_arr[index*NUM_NEIGHBORS +  8] = bend_spring_1;
		spring_arr[index*NUM_NEIGHBORS +  9] = bend_spring_2;
		spring_arr[index*NUM_NEIGHBORS + 10] = bend_spring_3;
		spring_arr[index*NUM_NEIGHBORS + 11] = bend_spring_4;
    }
}

//Generate randomized starting velocities in the XY plane
__global__
void generateRandomVelArray(int time, int N, glm::vec3 * arr)
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if(index < N)
    {
		glm::vec3 rand = float(SCENE_SCALE)*0.01f*(generateRandomNumberFromThread(time, index) - 0.5f);
        arr[index].x = 0.0;//rand.x;
        arr[index].y = 0.0;//rand.y;
        arr[index].z = 0.0;//rand.z;
    }
}

//TODO: Determine force between two bodies
__device__
glm::vec3 calculateAcceleration(glm::vec3 self_pos, glm::vec3 other_pos, float l0, float k)
{
    //    
    // F = -k(l - l0)
    //
    //        F     -k(l - l0)
    // a = ------ = ---------- 
    //     m_self     m_self
	
	glm::vec3 r((other_pos.x-self_pos.x), (other_pos.y-self_pos.y), (other_pos.z-self_pos.z));
	float l = sqrt(glm::dot(r,r));
	float m = POINT_MASS;

	if (l == 0)
		return glm::vec3(0,0,0);

	else {
		float f = k*(l - l0);
		glm::vec3 a = (f/m) * (r/l) * float(SCENE_SCALE) / 120.0f;
		return a;
	}
}

//TODO: Core force calc kernel global memory
__device__ 
glm::vec3 naiveAcc(int N, glm::vec3 my_pos, glm::vec3 * their_pos, glm::vec3 * springs)
{
	// current thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;

	// gravity
	glm::vec3 acc = glm::vec3(0.0f,0.0f, 2.0f / SCENE_SCALE);

	if (index < N)
	{
		// acceleration due to other particles
		for (int i = 0; i < 4; i++) {
			
			glm::vec3 spring_i = springs[index*NUM_NEIGHBORS + i];	// current spring object vector

			// ignore invalid neighbors
			int neighbor_index = spring_i.x;
			if (neighbor_index < 0) continue;
			
			glm::vec3 other_pos = their_pos[(int)spring_i.x];		// position of opposite particle
			float k = spring_i.y;									// spring constant k
			float l0 = spring_i.z;									// resting length l0

			// contribution from neighbor i
			acc += calculateAcceleration(my_pos, other_pos, l0, k);
		}
	}
	
	__syncthreads();

	// return the total acceleration
	return acc;
}


//TODO: Core force calc kernel shared memory
__device__ 
glm::vec3 sharedMemAcc(int N, glm::vec3 my_pos, glm::vec3 * their_pos, glm::vec3 * springs)
{
    glm::vec3 acc(0.0f,0.0f,0.0f);
	return acc;
}


//Simple Euler integration scheme
__global__
void updateF(int N, float dt, glm::vec3 * pos, glm::vec3 * acc, glm::vec3 * spr)
{
    int index = threadIdx.x + (blockIdx.x * blockDim.x);
    glm::vec3 my_pos;
    glm::vec3 accel(0,0,0);
	
    if(index < N) {
		my_pos = pos[index];
		accel = ACC(N, my_pos, pos, spr);
		acc[index] = accel;
	}

	__syncthreads();
}

__global__
void updateS(int N, float dt, glm::vec3 * pos, glm::vec3 * vel, glm::vec3 * acc)
{
    int index = threadIdx.x + (blockIdx.x * blockDim.x);
    if( index < N )
    {
        vel[index]   += acc[index]   * dt;
        pos[index].x += vel[index].x * dt;
        pos[index].y += vel[index].y * dt;
        pos[index].z += vel[index].z * dt;
    }
	
	__syncthreads();

	float distFromCenter = glm::distance(pos[index], glm::vec3(0));

	if (distFromCenter < SPHERE_RADIUS) {
		glm::vec3 normal = pos[index] - glm::vec3(0);
		glm::vec3 surfacePoint = normal/distFromCenter * float(SPHERE_RADIUS);
		pos[index] = surfacePoint;
	}

	__syncthreads();
}

//Update the vertex buffer object
//(The VBO is where OpenGL looks for the positions for the planets)
__global__
void sendToVBO(int N, glm::vec3 * pos, float * vbo, int width, int height, float s_scale)
{
    int index = threadIdx.x + (blockIdx.x * blockDim.x);

    float c_scale_w = -2.0f / s_scale;
    float c_scale_h = -2.0f / s_scale;
	float c_scale_z = -2.0f / s_scale;

    if(index<N)
    {
        vbo[4*index+0] = pos[index].x*c_scale_w;
        vbo[4*index+1] = pos[index].y*c_scale_h;
        vbo[4*index+2] = pos[index].z*c_scale_z;
        vbo[4*index+3] = 1;
    }
}

//Update the texture pixel buffer object
//(This texture is where openGL pulls the data for the height map)
__global__
void sendToPBO(int N, glm::vec3 * pos, float4 * pbo, int width, int height, float s_scale, glm::vec3 * springs)
{
    int index = threadIdx.x + (blockIdx.x * blockDim.x);
    int x = index % width;
    int y = index / width;
    float w2 = width / 2.0;
    float h2 = height / 2.0;

    float c_scale_w = width / s_scale;
    float c_scale_h = height / s_scale;

    glm::vec3 color(0.05, 0.15, 0.3);
    glm::vec3 acc = ACC(N, glm::vec3((x-w2)/c_scale_w,(y-h2)/c_scale_h,0), pos, springs);

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
    numParticles = N;
    dim3 fullBlocksPerGrid((int)ceil(float(numParticles)/float(blockSize)));

    cudaMalloc((void**)&dev_pos, numParticles*sizeof(glm::vec3));
    checkCUDAErrorWithLine("Kernel failed!");
    cudaMalloc((void**)&dev_vel, numParticles*sizeof(glm::vec3));
    checkCUDAErrorWithLine("Kernel failed!");
    cudaMalloc((void**)&dev_acc, numParticles*sizeof(glm::vec3));
    checkCUDAErrorWithLine("Kernel failed!");
    cudaMalloc((void**)&dev_spr, numParticles*NUM_NEIGHBORS*sizeof(glm::vec3));
    checkCUDAErrorWithLine("Kernel failed!");

    generateClothPosArray<<<fullBlocksPerGrid, blockSize>>>(numParticles, dev_pos);
    checkCUDAErrorWithLine("Kernel failed!");
    generateSpringArray<<<fullBlocksPerGrid, blockSize>>>(numParticles, dev_spr, dev_pos, 100.0, 100.0, 100.0);
    checkCUDAErrorWithLine("Kernel failed!");
    generateRandomVelArray<<<fullBlocksPerGrid, blockSize>>>(2, numParticles, dev_vel);
    checkCUDAErrorWithLine("Kernel failed!");
    cudaThreadSynchronize();
}

void cudaNBodyUpdateWrapper(float dt)
{
    dim3 fullBlocksPerGrid((int)ceil(float(numParticles)/float(blockSize)));
	updateF<<<fullBlocksPerGrid, blockSize>>>(numParticles, dt, dev_pos, dev_acc, dev_spr);
    checkCUDAErrorWithLine("Kernel failed!");
    updateS<<<fullBlocksPerGrid, blockSize>>>(numParticles, dt, dev_pos, dev_vel, dev_acc);
    checkCUDAErrorWithLine("Kernel failed!");
    cudaThreadSynchronize();
}

void cudaUpdateVBO(float * vbodptr, int width, int height)
{
    dim3 fullBlocksPerGrid((int)ceil(float(numParticles)/float(blockSize)));
    sendToVBO<<<fullBlocksPerGrid, blockSize>>>(numParticles, dev_pos, vbodptr, width, height, SCENE_SCALE);
    cudaThreadSynchronize();
}

void cudaUpdatePBO(float4 * pbodptr, int width, int height)
{
    dim3 fullBlocksPerGrid((int)ceil(float(width*height)/float(blockSize)));
	sendToPBO<<<fullBlocksPerGrid, blockSize, blockSize*sizeof(glm::vec4)>>>(numParticles, dev_pos, pbodptr, width, height, SCENE_SCALE, dev_spr);
    cudaThreadSynchronize();
}


