#include <stdio.h>
#include <cuda.h>
#include <cmath>
#include "glm/glm.hpp"
#include "utilities.h"
#include "kernel.h"
#include "boid.h"

#if SHARED == 1
    #define VEL(x,y,z,w) sharedMemVel(x,y,z,w)
    #define ACC(x,y,z) sharedMemAcc(x,y,z)
#else
    #define ACC(x,y,z) naiveAcc(x,y,z)
    #define VEL(x,y,z,w) naiveVel(x,y,z,w)
#endif

//GLOBALS
dim3 threadsPerBlock(blockSize);

int numObjects;
const __device__ float planetMass = 3e8;
const __device__ float starMass = 5e10;

const float scene_scale = 2e2; //size of the height map in simulation space

/*
0 - x coord
1 - y coord
2 - orientation
3 - vel magnitude
*/
glm::vec4 * dev_state;

//angularVel
float* dev_angularvel;

/*
0 - linear acc (force)
1 - angular acc (torque)
*/
glm::vec2 * dev_acc;

//FOR RK4
#if RK4==1
glm::vec4* dev_temppos;
glm::vec3* dev_tempvel;
glm::vec3* dev_accumvel;
glm::vec3* dev_accumaccel;
#endif

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

__global__
void setInitialState(int time, int N, glm::vec4* state, float* angVel, glm::vec2* acc, float scale)
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;

    if(index < N)
    {
        glm::vec3 rand1 = scale*(generateRandomNumberFromThread(time, index)-0.5f);
        glm::vec3 rand2 = generateRandomNumberFromThread(time+1, index);
		state[index].x = rand1.x;
        state[index].y = rand1.y;
        state[index].z = rand2.x*M_PI;//rand.z;
        state[index].w = rand2.y*MaxVelocity ;

		angVel[index] = 0;
		acc[index].x = 0;
		acc[index].y = 0;
    }
}

__device__
float clampAngle(float angle)
{

	while (angle > M_PI)
	{
		angle -= M2_PI;
	}
	while (angle < -M_PI)
	{
		angle += M2_PI;
	}

	return angle;
}

__device__
float getWorldOrientation(glm::vec2 V)
{
	float thetad = 0.0f;
	float vd = glm::length(V);
	if (vd < 0.0001)
	{
		thetad = 0.0;
	}
	else
	{
		if (abs(V.x) < 0.0001)
		{
			if (V.y > 0)
				thetad = M_PI / 2.0;
			else
				thetad = M_PI / -2.0;
		}
		else
			thetad = atan2(V.y, V.x);
	}
	
	thetad = clampAngle(thetad);

	return thetad;
}

__device__
glm::vec2 polarToWorld( glm::vec2 in)
{
	return in.y*glm::vec2( cos(in.x), sin(in.x));
}


__device__
glm::vec2 calculateArrivalVelocity( glm::vec2 myPos, glm::vec2 target)
{
		float dx =  target.x - myPos.x;
		float dy =  target.y - myPos.y;
		
		glm::vec2 e(dx,dy);
		float vDesired = KArrival*glm::length(e);
		return vDesired*e;
}

__device__
glm::vec2 calculateDepartureVelocity( glm::vec2 myPos, glm::vec2 target)
{
		float dx =  myPos.x - target.x;
		float dy =  myPos.y - target.y;
		
		glm::vec2 e(dx,dy);
		float eMagSq = 1e-30+dx*dx + dy*dy;
		float vDesired = KDeparture/eMagSq; 

		return vDesired*e;
}

__device__
glm::vec3 calculateAlignVelocity(glm::vec4 us, glm::vec4 them)
{
	glm::vec2 usPos( us.x, us.y);
	glm::vec2 themPos(them.x,them.y);

	float distSq = (them.x-us.x)*(them.x-us.x) + (them.y-us.y)*(them.y-us.y);
	if (distSq < RNeighborhoodSq)
	{
		glm::vec2 velWorld = polarToWorld( glm::vec2(them.z,them.w));
		return glm::vec3( velWorld.x, velWorld.y, 1);
	}
	return glm::vec3(0);
	
}


__device__
glm::vec3 calculateCohesionPosition(glm::vec4 us, glm::vec4 them)
{
	glm::vec2 usPos( us.x, us.y);
	glm::vec2 themPos(them.x,them.y);

	float distSq = (them.x-us.x)*(them.x-us.x) + (them.y-us.y)*(them.y-us.y);
	if (distSq < RNeighborhoodSq)
	{
		return glm::vec3( them.x, them.y, 1);
	}
	return glm::vec3(0);
	
}

__device__
glm::vec3 calculateSeparationVelocity(glm::vec4 us, glm::vec4 them)
{
	glm::vec2 usPos( us.x, us.y);
	glm::vec2 themPos(them.x,them.y);

	float distSq = (them.x-us.x)*(them.x-us.x) + (them.y-us.y)*(them.y-us.y);
	if (distSq < RNeighborhoodSq)
	{
		glm::vec2 dv =  calculateDepartureVelocity( glm::vec2(us.x,us.y) , glm::vec2(them.x,them.y));
		return glm::vec3( dv.x, dv.y, 1);
	}
	return glm::vec3(0);
	
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
    
	glm::vec3 usPos = glm::vec3(us.x,us.y,us.z);
	glm::vec3 themPos = glm::vec3(them.x,them.y,them.z);

	float distSq = 1e-30 + (usPos.x-themPos.x)*(usPos.x-themPos.x) 
		                 + (usPos.y-themPos.y)*(usPos.y-themPos.y)
						 + (usPos.z-themPos.z)*(usPos.z-themPos.z);

	return float(G)*them.w/distSq * (themPos - usPos) / sqrt(distSq);
	
}


//TODO: Core force calc kernel global memory
__device__ 
glm::vec3 naiveAcc(int N, glm::vec4 my_pos, glm::vec4 * their_pos)
{
	glm::vec3 acc = calculateAcceleration(my_pos, glm::vec4(0,0,0,starMass));
	//glm::vec3 acc(0.0f);
	for(int i=0; i<N; ++i)
	{
		acc+= calculateAcceleration(my_pos, glm::vec4(their_pos[i].x, their_pos[i].y,0,planetMass));
	}
    return acc;
}


//TODO: Core force calc kernel shared memory
__device__ 
glm::vec3 sharedMemAcc(int N, glm::vec4 my_pos, glm::vec4 * their_pos)
{
	const int tileSize = blockSize;
	float fractionTiles = (float)N/tileSize;

	int numberOfTiles =  fractionTiles -(int)fractionTiles>0?(int)fractionTiles+1:(int)fractionTiles;
	glm::vec3 acc = calculateAcceleration(my_pos, glm::vec4(0,0,0,starMass));
	
	extern __shared__ glm::vec4 sharedPos[];
	for(int tile=0; tile<numberOfTiles; ++tile)
	{
		int index = tile*tileSize+threadIdx.x;
		if(index<N)
		{
			sharedPos[threadIdx.x] = their_pos[index];
		}
		__syncthreads();

		for(int i=0;i<tileSize;i++)
		{
			acc+= calculateAcceleration(my_pos, glm::vec4(sharedPos[i].x, sharedPos[i].y,0,planetMass));
		}
		__syncthreads();
	}
    return acc;
}


//TODO: Core force calc kernel global memory
__device__ 
glm::vec2 naiveVel(int N, glm::vec4 my_pos, glm::vec4 * their_pos, Behavior behavior)
{
	glm::vec3 accum(0);
	//glm::vec3 acc(0.0f);
	for(int i=0; i<N; ++i)
	{
		if (behavior == Alignment)
		{
			glm::vec3 retVal = calculateAlignVelocity(my_pos, their_pos[i]);
			if (retVal.z != 0)
			{
				accum+= retVal;
			}
		}

		else if (behavior == Cohesion )
		{
			glm::vec3 retVal = calculateCohesionPosition(my_pos, their_pos[i]);
			if (retVal.z != 0)
			{
				accum+= retVal;
			}
		}

		else
		{
			glm::vec3 retVal = calculateSeparationVelocity(my_pos, their_pos[i]);
			if (retVal.z != 0)
			{
				accum+= retVal;
			}
		}

	}

	glm::vec2 average = glm::vec2(accum.x, accum.y) / accum.z;
	
	if( behavior == Alignment)
	{
		return KAlign*glm::normalize(average);
	}
	else if(behavior == Cohesion)
	{
		glm::vec2 e(average.x-my_pos.x,average.y-my_pos.y);
		return KCohesion*e;
	}
	else if(behavior == Separation)
	{
		return KSeparate*average;
	}

	else
		return glm::vec2(0);
}


//TODO: Core force calc kernel shared memory
__device__ 
glm::vec2 sharedMemVel(int N, glm::vec4 my_pos, glm::vec4 * their_pos, Behavior behavior)
{
	const int tileSize = blockSize;
	float fractionTiles = (float)N/tileSize;

	int numberOfTiles =  fractionTiles -(int)fractionTiles>0?(int)fractionTiles+1:(int)fractionTiles;
	glm::vec3 accum(0);
	
	__shared__ glm::vec4 sharedPos[blockSize];
	for(int tile=0; tile<numberOfTiles; ++tile)
	{
		int index = tile*tileSize+threadIdx.x;
		if(index<N)
		{
			sharedPos[threadIdx.x] = their_pos[index];
		}
		__syncthreads();

		for(int i=0;i<tileSize;i++)
		{
			if (behavior == Alignment)
			{
				glm::vec3 retVal = calculateAlignVelocity(my_pos, their_pos[i]);
				if (retVal.z != 0)
				{
					accum+= retVal;
				}
			}

			else if (behavior == Cohesion )
			{
				glm::vec3 retVal = calculateCohesionPosition(my_pos, their_pos[i]);
				if (retVal.z != 0)
				{
					accum+= retVal;
				}
			}

			else
			{
				glm::vec3 retVal = calculateSeparationVelocity(my_pos, their_pos[i]);
				if (retVal.z != 0)
				{
					accum+= retVal;
				}
			}
		}
		__syncthreads();
	}

	glm::vec2 average = glm::vec2(accum.x, accum.y) / accum.z;
	
	if( behavior == Alignment)
	{
		return KAlign*glm::normalize(average);
	}
	else if(behavior == Cohesion)
	{
		glm::vec2 e(average.x-my_pos.x,average.y-my_pos.y);
		return KCohesion*e;
	}
	else if(behavior == Separation)
	{
		return KSeparate*average;
	}

	else
		return glm::vec2(0);
}

//Simple Euler integration scheme
__global__
void updateF(int N, float dt, glm::vec4 * state,float* angVel, glm::vec2 * acc, Behavior behavior)
{
    int index = threadIdx.x + (blockIdx.x * blockDim.x);
    glm::vec4 my_state;
	float myAngVel;
    glm::vec2 worldVel;

    if(index < N) 
	{
		my_state = state[index];
		myAngVel = angVel[index];
	}

	if ( behavior == Alignment || behavior == Cohesion || behavior == Separation)
		worldVel = VEL(N, my_state, state, behavior);
	else if (behavior == Flocking)
	{
		glm::vec2 aVel = VEL(N, my_state, state, Alignment);
		glm::vec2 cVel = VEL(N, my_state, state, Cohesion);
		glm::vec2 sVel = VEL(N, my_state, state, Separation);

		worldVel = 2.0f*sVel + 20.0f*cVel + 0.6f*aVel;
	}
	else if ( behavior == Arrival)
		worldVel = calculateArrivalVelocity( glm::vec2(my_state.x,my_state.y), glm::vec2(0));

	else if (behavior == Departure)
		worldVel = calculateDepartureVelocity( glm::vec2(my_state.x,my_state.y), glm::vec2(0));

	float thetaDesired = getWorldOrientation( worldVel);
	float vDesired = glm::length(worldVel);
	clamp(vDesired,0.0f,MaxVelocity);

    if(index < N)
	{
		float force = Kv0*(vDesired - my_state.w);
		clamp(force,0.0f,MaxForce);
		float torque = ( -Kv1*myAngVel - Kp1*my_state.z + Kp1*thetaDesired);
		clamp(torque,0.0f,MaxTorque);
		acc[index].x = force;
		acc[index].y = torque;

	}
}

__global__
void updateS(int N, float dt, glm::vec4 * state, float* angVel, glm::vec2* acc)
{
    int index = threadIdx.x + (blockIdx.x * blockDim.x);
    if( index < N )
    {
		glm::vec2 velWorldSpace = state[index].w*glm::vec2(cos(state[index].z), sin(state[index].z) );

		state[index].x += velWorldSpace.x*dt;
		state[index].y += velWorldSpace.y*dt;
		state[index].z += angVel[index]*dt;
		state[index].w += acc[index].x*dt;
		angVel[index]  += acc[index].y*dt;
    }
}

__global__
void RK4Step(int N, float rkStep, float accumMultiplier,glm::vec4* pos,glm::vec3* vel,glm::vec3* accel, glm::vec4* tempPos, glm::vec3* tempVel, glm::vec3* accumVel, glm::vec3*  accumAccel)
{
    int index = threadIdx.x + (blockIdx.x * blockDim.x);
    if( index < N )
    {
        glm::vec4 my_pos = tempPos[index];
		glm::vec3 acc = accel[index];
		accumAccel[index] += accumMultiplier*acc;
		accumVel[index] += accumMultiplier*tempVel[index];

		tempVel[index] = vel[index] + rkStep* acc;
        tempPos[index].x = pos[index].x + tempVel[index].x * rkStep;
		tempPos[index].y = pos[index].y + tempVel[index].y * rkStep;
        tempPos[index].z = pos[index].z + tempVel[index].z * rkStep;		
	}
}

__global__
void RK4FinalUpdate(int N, float dt,glm::vec4* pos,glm::vec3* vel, glm::vec3* accumVel, glm::vec3* accumAccel)
{
    int index = threadIdx.x + (blockIdx.x * blockDim.x);
    if( index < N )
    {
		float oneSixthInverse = dt/6.0f;
        vel[index]   += oneSixthInverse*accumAccel[index];
        pos[index].x += oneSixthInverse*accumVel[index].x;
		pos[index].y += oneSixthInverse*accumVel[index].y;
		pos[index].z += oneSixthInverse*accumVel[index].z;
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
        vbo[4*index+2] = pos[index].z;
        vbo[4*index+3] = pos[index].w;
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
    dim3 fullBlocksPerGrid((int)ceil(float(N)/float(blockSize)));

    cudaMalloc((void**)&dev_state, N*sizeof(glm::vec4));
    checkCUDAErrorWithLine("Kernel failed!");
    cudaMalloc((void**)&dev_angularvel, N*sizeof(float));
    checkCUDAErrorWithLine("Kernel failed!");
    cudaMalloc((void**)&dev_acc, N*sizeof(glm::vec2));
    checkCUDAErrorWithLine("Kernel failed!");

    //generateRandomPosArray<<<fullBlocksPerGrid, blockSize>>>(1, numObjects, dev_state, scene_scale, planetMass);
    //checkCUDAErrorWithLine("Kernel failed!");
    //generateCircularVelArray<<<fullBlocksPerGrid, blockSize>>>(2, numObjects, dev_vel, dev_state);
    //checkCUDAErrorWithLine("Kernel failed!");

	setInitialState<<<fullBlocksPerGrid, blockSize>>>(1, numObjects, dev_state, dev_angularvel, dev_acc, scene_scale);
    checkCUDAErrorWithLine("Kernel failed!");

#if RK4 == 1
	//FOR RK4
    cudaMalloc((void**)&dev_temppos, N*sizeof(glm::vec4));
    checkCUDAErrorWithLine("Kernel failed!");
    cudaMalloc((void**)&dev_tempvel, N*sizeof(glm::vec3));
    checkCUDAErrorWithLine("Kernel failed!");
    cudaMalloc((void**)&dev_accumaccel, N*sizeof(glm::vec3));
    checkCUDAErrorWithLine("Kernel failed!");
    cudaMalloc((void**)&dev_accumvel, N*sizeof(glm::vec3));
    checkCUDAErrorWithLine("Kernel failed!");
#endif
    cudaThreadSynchronize();
}

void cudaNBodyUpdateWrapper(float dt, Behavior mode)
{
    dim3 fullBlocksPerGrid((int)ceil(float(numObjects)/float(blockSize)));
#if RK4 == 1
	cudaMemcpy( dev_temppos, dev_state, numObjects*sizeof(glm::vec4),cudaMemcpyDeviceToDevice);
	cudaMemcpy( dev_tempvel, dev_vel, numObjects*sizeof(glm::vec3),cudaMemcpyDeviceToDevice);
	//RK4
	//First step
    updateF<<<fullBlocksPerGrid, blockSize, blockSize*sizeof(glm::vec4)>>>(numObjects, dt, dev_temppos, dev_vel, dev_acc);
	RK4Step<<<fullBlocksPerGrid, blockSize>>>(numObjects, dt/2.0f, 1.0f, dev_state, dev_vel,dev_acc,dev_temppos,dev_tempvel,dev_accumvel,dev_accumaccel);
	//Second step
    updateF<<<fullBlocksPerGrid, blockSize, blockSize*sizeof(glm::vec4)>>>(numObjects, dt, dev_temppos, dev_vel, dev_acc);
	RK4Step<<<fullBlocksPerGrid, blockSize>>>(numObjects, dt/2.0f, 2.0f, dev_state, dev_vel,dev_acc,dev_temppos,dev_tempvel,dev_accumvel,dev_accumaccel);
	//Third step
    updateF<<<fullBlocksPerGrid, blockSize, blockSize*sizeof(glm::vec4)>>>(numObjects, dt, dev_temppos, dev_vel, dev_acc);
	RK4Step<<<fullBlocksPerGrid, blockSize>>>(numObjects, dt, 2.0f, dev_state, dev_vel,dev_acc,dev_temppos,dev_tempvel,dev_accumvel,dev_accumaccel);
	//Fourth step
    updateF<<<fullBlocksPerGrid, blockSize, blockSize*sizeof(glm::vec4)>>>(numObjects, dt, dev_temppos, dev_vel, dev_acc);
	RK4Step<<<fullBlocksPerGrid, blockSize>>>(numObjects, dt, 1.0f, dev_state, dev_vel,dev_acc,dev_temppos,dev_tempvel,dev_accumvel,dev_accumaccel);

	RK4FinalUpdate<<<fullBlocksPerGrid, blockSize>>>(numObjects, dt, dev_state, dev_vel,dev_accumvel,dev_accumaccel);
	cudaMemset(dev_accumaccel,0,numObjects*sizeof(glm::vec3));
	cudaMemset(dev_accumvel,0,numObjects*sizeof(glm::vec3));
#else
    updateF<<<fullBlocksPerGrid, blockSize, blockSize*sizeof(glm::vec4)>>>(numObjects, dt, dev_state, dev_angularvel,dev_acc, mode);
    checkCUDAErrorWithLine("Kernel failed!");
    updateS<<<fullBlocksPerGrid, blockSize>>>(numObjects, dt, dev_state, dev_angularvel, dev_acc);
	checkCUDAErrorWithLine("Kernel failed!");
#endif
    cudaThreadSynchronize();
}

void cudaUpdateVBO(float * vbodptr, int width, int height)
{
    dim3 fullBlocksPerGrid((int)ceil(float(numObjects)/float(blockSize)));
    sendToVBO<<<fullBlocksPerGrid, blockSize>>>(numObjects, dev_state, vbodptr, width, height, scene_scale);
    cudaThreadSynchronize();
}

void cudaUpdatePBO(float4 * pbodptr, int width, int height)
{
    dim3 fullBlocksPerGrid((int)ceil(float(width*height)/float(blockSize)));
    sendToPBO<<<fullBlocksPerGrid, blockSize, blockSize*sizeof(glm::vec4)>>>(numObjects, dev_state, pbodptr, width, height, scene_scale);
    cudaThreadSynchronize();
}


