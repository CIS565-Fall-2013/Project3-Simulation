#include <stdio.h>
#include <cuda.h>
#include <cmath>
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "utilities.h"
#include "kernel.h"

#if SHARED == 1
    //#define ACC(x,y,z) sharedMemAcc(x,y,z)
	//#define ACC(x,y,z) prefetchAcc(x,y,z)
	#define ACC(x,y,z) unrolledAcc(x,y,z)
#else
    #define ACC(x,y,z) naiveAcc(x,y,z)
#endif

//GLOBALS
//dim3 threadsPerBlock(blockSize);

#define MAX_VEL 1.0f

int numObjects;
const float planetMass = 1;
//const __device__ float starMass = 5e10;
const __device__ float starMass = 0;
float totalTime = 0;
int timeCount = 0;

const float scene_scale = 2e2; //size of the height map in simulation space

glm::vec4 * dev_pos;
glm::vec3 * dev_vel;
glm::vec3 * dev_acc;
float* dev_rot; //rotation of each boid
float* dev_angular_vel; //angular velocity of each boid

#define PROP_GAIN 16.0f
#define DERIV_GAIN 8.0f

#define TORQUE_PROP 16.0f
#define TORQUE_DERIV 8.0f

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
    thrust::default_random_engine rng(hash((index+1)*(time+1)));
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
    glm::vec3 us3(us);
	glm::vec3 them3(them);
	float rSquared = glm::distance2(us3, them3) /* + 1*/;

	/*if(rSquared < RSQUARED_CUTOFF)
		return glm::vec3(0, 0, 0);*/

	float m_them = them.w;
	glm::vec3 dir = them3 - us3;
	float mag = (G*(m_them / rSquared));
	float dirLen = glm::length(dir);
	if( dirLen < PHYS_EPSILON )
		return glm::vec3(0, 0, 0);
	else
		return mag*dir/dirLen;
}

//the below function applies a force to push the boids in to the right place.
__device__ 
glm::vec3 apply_control_force(glm::vec4 my_pos, glm::vec3 desired_vel, glm::vec3 curr_vel)
{
	return DERIV_GAIN * (desired_vel - curr_vel);
}

//the below function applies a torque to make the boids face the correct direction
//__device__ 
//glm::vec3 apply_control_torque(float desired_angle, float curr_angle)
//{
//	//Assuming inertia is 1
//	float inertia = 1;
//	return inertia * (-TORQUE_DERIV * 
//}

//TODO: Core force calc kernel global memory
__device__ 
glm::vec3 naiveAcc(int N, glm::vec4 my_pos, glm::vec4 * their_pos)
{
	glm::vec3 acc;
	acc = acc + calculateAcceleration(my_pos, glm::vec4(0,0,0,starMass));	

	for(int i = 0; i < N; i++){
		 acc = acc + calculateAcceleration(my_pos, their_pos[i]);
	}
    return acc;
}

__device__ 
glm::vec3 sharedMemAcc(int N, glm::vec4 my_pos, glm::vec4 * their_pos)
{
	extern __shared__ glm::vec4 sharedPos[];
	int tx = threadIdx.x;
	int index = threadIdx.x + (blockIdx.x * blockDim.x);
	glm::vec3 acc;
	acc = acc + calculateAcceleration(my_pos, glm::vec4(0,0,0,starMass));

	for(int j = 0; j < N/TILE_SIZE; j++){
		//read tile into shared memory
		for(int i = 0; i < TILE_SIZE; i++){
			sharedPos[i] = their_pos[i + j*TILE_SIZE]; 
		}
		__syncthreads(); //everything must be in shared before we can proceed
		#pragma unroll TILE_SIZE
		for(int i = 0; i < TILE_SIZE; i++){
			 acc = acc + calculateAcceleration(my_pos, sharedPos[i]);
		}
		__syncthreads(); 
	}
    return acc;
}

__device__ 
glm::vec3 prefetchAcc(int N, glm::vec4 my_pos, glm::vec4 * their_pos)
{
	extern __shared__ glm::vec4 sharedPos[];
	int tx = threadIdx.x;
	int index = threadIdx.x + (blockIdx.x * blockDim.x);
	glm::vec3 acc;
	acc = acc + calculateAcceleration(my_pos, glm::vec4(0,0,0,starMass));

	//load first tile into registers
	glm::vec4 nextTile[TILE_SIZE];

	for(int i = 0; i < TILE_SIZE; i++){
		nextTile[i] = their_pos[i]; 
	}
	for(int j = 0; j < NUM_TILES; j++){
		//deposit registers into shared memory
		for(int i = 0; i < TILE_SIZE; i++){
			sharedPos[i] = nextTile[i]; 
		}
		__syncthreads(); 
		//Load next tile into registers
		for(int i = 0; i < TILE_SIZE; i++){
			//when j == numTiles-1, then there is no j+1th tile.
			nextTile[i] = their_pos[i + min((j+1),NUM_TILES-1)*TILE_SIZE]; 
		}
		//Accumulate acceleration
		for(int i = 0; i < TILE_SIZE; i++){
			 acc = acc + calculateAcceleration(my_pos, sharedPos[i]);
		}
		__syncthreads(); 
	}
    return acc;
}

__device__ 
glm::vec3 unrolledAcc(int N, glm::vec4 my_pos, glm::vec4 * their_pos)
{
	extern __shared__ glm::vec4 sharedPos[];
	int tx = threadIdx.x;
	int index = threadIdx.x + (blockIdx.x * blockDim.x);
	glm::vec3 acc;
	acc = acc + calculateAcceleration(my_pos, glm::vec4(0,0,0,starMass));

	//load first tile into registers
	glm::vec4 nextTile[TILE_SIZE];

	#pragma unroll TILE_SIZE
	for(int i = 0; i < TILE_SIZE; i++){
		nextTile[i] = their_pos[i]; 
	}

	//#pragma unroll NUM_TILES
	for(int j = 0; j < NUM_TILES; j++){
		//deposit registers into shared memory

		#pragma unroll TILE_SIZE
		for(int i = 0; i < TILE_SIZE; i++){
			sharedPos[i] = nextTile[i]; 
		}
		__syncthreads(); 
		//Load next tile into registers
		#pragma unroll TILE_SIZE
		for(int i = 0; i < TILE_SIZE; i++){
			//when j == numTiles-1, then there is no j+1th tile.
			nextTile[i] = their_pos[i + min((j+1),NUM_TILES-1)*TILE_SIZE]; 
		}
		//Accumulate acceleration
		#pragma unroll TILE_SIZE
		for(int i = 0; i < TILE_SIZE; i++){
			 acc = acc + calculateAcceleration(my_pos, sharedPos[i]);
		}
		__syncthreads(); 
	}
    return acc;
}

__device__ glm::vec3 calculateReflectionDirection(glm::vec3 normal, glm::vec3 incident) {
  const float cosI = -glm::dot( normal, incident );
  return glm::normalize(incident + 2 * cosI * normal);
}

__device__ glm::vec3 resolveCollisions(int N, glm::vec4 my_pos, glm::vec4 * their_pos, glm::vec3 my_vel)
{
	glm::vec3 my_pos3(my_pos);
	int index = threadIdx.x + (blockIdx.x * blockDim.x);
	//glm::vec3 reflectV = glm::vec3(0);//my_vel; //keep the same velocity, if no collision
	glm::vec3 reflectV = my_vel;
	for(int i = 0; i < N; i++){
		if( i != index ){ //don't collide with yourself!
			glm::vec3 their_pos3(their_pos[i]);
			float dist = glm::distance(my_pos3, their_pos3);
			if( dist < COLLISION_RAD ){ //collision!
				glm::vec3 normal = glm::normalize(my_pos3 - their_pos3); //normal is pointing away from "them", towards us.
				glm::vec3 newPos = COLLISION_RAD*0.5f*normal + 0.5f*(their_pos3 + my_pos3);
				their_pos[index].x = newPos.x;
				their_pos[index].y = newPos.y;
				their_pos[index].z = newPos.z;
				reflectV = glm::length(my_vel)*glm::normalize(calculateReflectionDirection(my_vel, normal));
			}
		}
	}
	return reflectV;
}

__device__ glm::vec3 seek(glm::vec3 my_pos, glm::vec3 goal_pos)
{
	glm::vec3 eGlob = goal_pos - my_pos; //Global error vector
	return MAX_VEL * glm::normalize(eGlob);
}

__device__ glm::vec3 arrival(glm::vec3 my_pos, glm::vec3 goal_pos)
{
	glm::vec3 eGlob = goal_pos - my_pos; //Global error vector
	float arrivalGain = 0.1f; //proportional gain.
	glm::vec3 vArrival = arrivalGain*eGlob;

	if ( glm::length(vArrival) >= 0.1f){
		return vArrival;
	}
	else{
		return glm::vec3(0,0,0);
	}
}



__device__ glm::vec3 separation(int N, glm::vec3 my_pos, glm::vec3 goal_pos, glm::vec4* their_pos)
{
	int index = threadIdx.x + (blockIdx.x * blockDim.x);
	float RNeighborhood = 800.0f;
	float vGain = 1.0f;
	glm::vec3 vSeparate(0, 0, 0);
	for(int i = 0; i < N; i++){
		if(i != index){
			//di is the vector from agent i to this agent
			//remember, we want to move AWAY from the other agents.
			glm::vec3 theirPos3 = glm::vec3(their_pos[i]);
			glm::vec3 di = my_pos - theirPos3;
			//assuming weight wi is 1.
			float softening_term = 1; //used to prevent singularity when particles are too close together
			if( glm::length(di) < RNeighborhood && glm::length(di) >= EPSILON)
				vSeparate = vSeparate + (di/(glm::length2(di) + softening_term));
			else if(glm::length(di) < EPSILON){
				float randGain = 0.01f;
				glm::vec3 randVec = generateRandomNumberFromThread(1337, index);
				vSeparate = vSeparate + randGain * randVec;
			}
		}
	}
	return vGain * vSeparate;
}

__device__ glm::vec3 leader_follow(int N, glm::vec3 my_pos, glm::vec3 goal_pos, glm::vec4* their_pos)
{
	float cSep = 1.5;
	float cArrive = 1;
	return cSep*separation(N, my_pos, goal_pos, their_pos) + cArrive*arrival(my_pos, goal_pos);
}

//Simple Euler integration scheme
__global__
void updateF(int N, float dt, glm::vec4 * pos, glm::vec3 * vel, glm::vec3 * acc, BehaviorType bType)
{
	
    int index = threadIdx.x + (blockIdx.x * blockDim.x);

    glm::vec4 my_pos;
	glm::vec3 my_vel;
    glm::vec3 accel;
	glm::vec3 newVel;
	glm::vec3 desired_vel;
	glm::vec3 goal_pos = glm::vec3(pos[0]);
	glm::vec3 leader_vel = vel[0];

    if(index < N){
		my_pos = pos[index];
		my_vel = vel[index];
		if(bType == SEEK){
			desired_vel = seek(glm::vec3(my_pos), goal_pos);
		} else if(bType == ARRIVAL){
			desired_vel = arrival(glm::vec3(my_pos), goal_pos);
		} else if(bType == SEPARATION){
			desired_vel = separation(N, glm::vec3(my_pos), goal_pos, pos);
		} else if(bType == LEADER){
			desired_vel = leader_follow(N, glm::vec3(my_pos), goal_pos - 150.0f*leader_vel, pos);
		}
	}

	if(index == 0){
		desired_vel = glm::vec3(0, 0, 0);
	}

    //accel = ACC(N, my_pos, pos);
	float mass = my_pos.w;
	accel = (1.0f/mass)*apply_control_force(my_pos, desired_vel, my_vel);
	//newVel = resolveCollisions(N, my_pos, pos, my_vel);

    if(index < N){
		acc[index] = accel;
		vel[index] = my_vel;
		//vel[index] = newVel;
	}
}

__global__
void updateS(int N, float dt, glm::vec4 * pos, glm::vec3 * vel, glm::vec3 * acc)
{
    int index = threadIdx.x + (blockIdx.x * blockDim.x);
    if( index < N )
    {
		if(INTEGRATION_TYPE == 0){ //Symplectic Euler
				vel[index]   += acc[index]   * dt;
				pos[index].x += vel[index].x * dt;
				pos[index].y += vel[index].y * dt;
				pos[index].z += vel[index].z * dt;
		}else if ( INTEGRATION_TYPE == 1 ){ //Forward Euler
				pos[index].x += vel[index].x * dt;
				pos[index].y += vel[index].y * dt;
				pos[index].z += vel[index].z * dt;
				vel[index]   += acc[index]   * dt;
		}
	}
}

//Update the vertex buffer object
//(The VBO is where OpenGL looks for the positions for the planets)
__global__
void sendToVBO(int N, glm::vec4 * pos, float * vbo, int width, int height, float s_scale, float* rotation)
{
    int index = threadIdx.x + (blockIdx.x * blockDim.x);

    float c_scale_w = -2.0f / s_scale;
    float c_scale_h = -2.0f / s_scale;

    if(index<N)
    {
        vbo[4*index+0] = pos[index].x*c_scale_w;
        vbo[4*index+1] = pos[index].y*c_scale_h;
        vbo[4*index+2] = 0;
        //vbo[4*index+3] = ((float)index)/N * TWO_PI;
		vbo[4*index+3] = (rotation[index])/N * TWO_PI;
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

void cleanupCuda(){
	cudaDeviceReset();
}
//Initialize memory, update some globals
glm::vec4* initCuda(int N, int blockSize)
{
    numObjects = N;
    dim3 fullBlocksPerGrid((int)ceil(float(N)/float(blockSize)));

    cudaMalloc((void**)&dev_pos, N*sizeof(glm::vec4));
    checkCUDAErrorWithLine("Kernel failed!");
    cudaMalloc((void**)&dev_vel, N*sizeof(glm::vec3));
    checkCUDAErrorWithLine("Kernel failed!");
    cudaMalloc((void**)&dev_acc, N*sizeof(glm::vec3));
    checkCUDAErrorWithLine("Kernel failed!");


	cudaMalloc((void**)&dev_rot, N*sizeof(float));
	float* initialRotations = new float[N];
	for(int i = 0; i < N; i++){
		initialRotations[i] = 0.0f;
	}
	checkCUDAErrorWithLine("Kernel failed!");
	cudaMemcpy(dev_rot, initialRotations, N*sizeof(float), cudaMemcpyHostToDevice);
	delete initialRotations;

	cudaMalloc((void**)&dev_angular_vel, N*sizeof(float));
	float* initialAngVel = new float[N];
	for(int i = 0; i < N; i++){
		initialAngVel[i] = 0.0f;
	}
    checkCUDAErrorWithLine("Kernel failed!");
	cudaMemcpy(dev_angular_vel, initialAngVel, N*sizeof(float), cudaMemcpyHostToDevice);
	delete initialAngVel;

    generateRandomPosArray<<<fullBlocksPerGrid, blockSize>>>(1, numObjects, dev_pos, scene_scale, planetMass);
    checkCUDAErrorWithLine("Kernel failed!");
    generateCircularVelArray<<<fullBlocksPerGrid, blockSize>>>(2, numObjects, dev_vel, dev_pos);
    checkCUDAErrorWithLine("Kernel failed!");
    cudaThreadSynchronize();

	return dev_pos;

	//atexit(cleanupCuda);
}

void cudaNBodyUpdateWrapper(float dt, int blockSize, BehaviorType bType)
{
    dim3 fullBlocksPerGrid((int)ceil(float(numObjects)/float(blockSize)));

	cudaEvent_t start, stop;
	float nathanTime;
	cudaEventCreate(&start);
	cudaEventRecord(start,0);
    updateF<<<fullBlocksPerGrid, blockSize, TILE_SIZE*sizeof(glm::vec4)>>>(numObjects, dt, dev_pos, dev_vel, dev_acc, bType);
	cudaEventCreate(&stop);
	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&nathanTime, start,stop);
	/*printf("Elapsed time : %f ms\n",nathanTime);*/
	totalTime = totalTime + nathanTime;
	timeCount++;
	if(timeCount >= 100){
		printf("Avg kernel time : %f ms\n",totalTime/100);
		totalTime = 0;
		timeCount = 0;
	}

    checkCUDAErrorWithLine("Kernel failed!");

    updateS<<<fullBlocksPerGrid, blockSize>>>(numObjects, dt, dev_pos, dev_vel, dev_acc);
    checkCUDAErrorWithLine("Kernel failed!");
    cudaThreadSynchronize();
}

void cudaUpdateVBO(float * vbodptr, int width, int height, int blockSize)
{
    dim3 fullBlocksPerGrid((int)ceil(float(numObjects)/float(blockSize)));
    sendToVBO<<<fullBlocksPerGrid, blockSize>>>(numObjects, dev_pos, vbodptr, width, height, scene_scale, dev_rot);
    cudaThreadSynchronize();
}

void cudaUpdatePBO(float4 * pbodptr, int width, int height, int blockSize)
{
    dim3 fullBlocksPerGrid((int)ceil(float(width*height)/float(blockSize)));
    sendToPBO<<<fullBlocksPerGrid, blockSize, TILE_SIZE*sizeof(glm::vec4)>>>(numObjects, dev_pos, pbodptr, width, height, scene_scale);
    cudaThreadSynchronize();
}


