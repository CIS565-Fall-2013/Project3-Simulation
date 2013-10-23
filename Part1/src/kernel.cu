#include <stdio.h>
#include <cuda.h>
#include <cmath>
#include "glm/glm.hpp"
#include "utilities.h"
#include "kernel.h"

//GLOBALS
dim3 threadsPerBlock(blockSize);

#define	CAMHEIGHT	75.0f
#define CAMFORWARD	30.0f

int numObjects;
const float planetMass = 3e8;
const __device__ float starMass = 5e10;
const __device__ float GravConst = 6.67384e-11;
__device__ bool prefetch;
__device__ int attachedToIndex = 0;
const float scene_scale = 2e2; //size of the height map in simulation space
bool camUpdate = false;

#if SHARED == 1
//	if (prefetch)
		#define ACC(x,y,z) pfSharedMemAcc(x,y,z)
//	else
//		#define ACC(x,y,z) sharedMemAcc(x,y,z)
	#define FLOCK(p,q,r,s,t) FlockGlobal(p,q,r,s,t)
#else
    #define ACC(x,y,z) naiveAcc(x,y,z)
	#define FLOCK(p,q,r,s,t) FlockGlobal(p,q,r,s,t)
#endif

glm::vec4 * dev_pos;
glm::vec3 * dev_vel;
glm::vec3 * dev_acc;
glm::vec4 * dev_campos;

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
		std::cin.get ();
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

// Calculate gravitational acceleration.
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

//Simple Euler integration scheme
__global__
void updateS(int N, float dt, glm::vec4 * pos, glm::vec3 * vel, glm::vec3 * acc, glm::vec4 * dev_campos, bool cameraUpdate)
{
    int index = threadIdx.x + (blockIdx.x * blockDim.x);
    if( index < N )
    {
		bool isCameraAttached = false;
		glm::vec4 curPos = pos [index];
		if (cameraUpdate)
		{
			if (isApproximately (curPos.x, dev_campos->x) &&
				isApproximately (curPos.y, dev_campos->y+CAMFORWARD) &&
				isApproximately (curPos.z, dev_campos->z-CAMHEIGHT))
				isCameraAttached = true;
		}

        vel[index]   += acc[index]   * dt;
        pos[index].x += vel[index].x * dt;
        pos[index].y += vel[index].y * dt;
        pos[index].z += vel[index].z * dt;

		if (isCameraAttached)
		{
			dev_campos->z = curPos.z + CAMHEIGHT;
			dev_campos->y = curPos.y - CAMFORWARD;
		}
    }
}

//TODO: Done!
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
	glm::vec3 forceDir = glm::vec3 (them.x - us.x, them.y - us.y, them.z - us.z);
	float dist = sqrt (glm::dot (forceDir, forceDir));
	
	if (dist > 0)
	{
		forceDir /= dist;	// Force direction is now normalized and we have distance between the two objects (r)!
		float accVal = (GravConst * them.w) / (dist*dist);
		return forceDir * accVal;
	}

	return glm::vec3 (0);
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

// Calculate acceleration for each object using global memory, given the number of particles (N), 
// their positions (their_pos) and this object's position (my_pos).
// Written by Rohith Chandran.
__device__ 
glm::vec3 naiveAcc(int N, glm::vec4 my_pos, glm::vec4 * their_pos)
{
	glm::vec3 acc = glm::vec3 (0);
	int index = threadIdx.x + (blockIdx.x * blockDim.x);

	for (int i = 0; i < N; i ++)
	{	
		if (isApproximately (their_pos [i].x, my_pos.x) && 
			isApproximately (their_pos [i].y, my_pos.y) && 
			isApproximately (their_pos [i].z, my_pos.z))
				continue;
		acc += calculateAcceleration(my_pos, their_pos [i]);
	}
	acc += calculateAcceleration (my_pos, glm::vec4 (0, 0, 0, starMass));

	return acc;
}

// For float comparisons.
// Written by Rohith Chandran.
__device__ bool isApproximately (const float &a, const float &b)
{
	if ((a >= (b - 0.001)) && (a <= (b + 0.001)))
		return true;
	return false;
}

// Calculate acceleration for each object using shared memory, given the number of particles (N), 
// their positions (their_pos) and this object's position (my_pos).
// Written by Rohith Chandran.
__device__ 
glm::vec3 sharedMemAcc(int N, glm::vec4 my_pos, glm::vec4 * their_pos)
{
	extern __shared__ glm::vec4 shared_pos [];
	int threadNo = blockDim.x * blockIdx.x + threadIdx.x;

	glm::vec3 acc = glm::vec3 (0);

	// Loop over each block (assuming parallelization of objects) and load objects from global to shared memory.
	// Each block of threads will load blockDim.x no. of objects from global memory to shared memory; 
	// Thus, we load the entire set of positions in global memory into shared memory iteratively, one block at a time.
	for (int j = 0; j < ceil (N / (float)blockDim.x); j ++)
	{
		// Calculate global memory index that should be accessed by this thread.
		int index = blockDim.x * j + threadIdx.x;
		// Load the value from global to shared. 
		if (index < N)
			shared_pos [threadIdx.x] = their_pos [index];

//		Synchronize here.
		__syncthreads();		

		// Compute acceleration for all threads.
		// Loop over each object, and calculate acceleration.
		for (int i = 0; i < blockDim.x; i ++)
		{	
			// If the block of global memory we're loading into shared mem corresponds to the last block in the grid, 
			// it can contain less than blockDim.x elements. In such a situation, break out of the loop once we pass 
			// the last element in that "block".
			if (j == (floor (N / (float)blockDim.x)))
				if (i >= (N%blockDim.x))
					break;

			// A body cannot exert a force on itself, so skip..
			if (isApproximately (shared_pos [i].x, my_pos.x) && 
				isApproximately (shared_pos [i].y, my_pos.y) && 
				isApproximately (shared_pos [i].z, my_pos.z))
				continue;

			acc += calculateAcceleration(my_pos, shared_pos [i]);
		}
	}

	// Calculate acceleration due to star.
	acc += calculateAcceleration (my_pos, glm::vec4 (0, 0, 0, starMass));
	
	return acc;
}

// Shared memory acceleration calculation with prefetching.
// Written as a separate function to compare performance.
// Written by Rohith Chandran.
__device__ 
glm::vec3 pfSharedMemAcc(int N, glm::vec4 my_pos, glm::vec4 * their_pos)
{
	extern __shared__ glm::vec4 shared_pos [];
	int threadNo = blockDim.x * blockIdx.x + threadIdx.x;

	glm::vec3 acc = glm::vec3 (0);
	glm::vec4 prefetcher = glm::vec4 (0);

	int index = threadIdx.x;
	if (index < N)
		prefetcher = their_pos [index];		// Prefetch first element into register.

	for (int j = 0; j < ceil (N / (float)blockDim.x); j ++)
	{
		index = blockDim.x * (j+1) + threadIdx.x;

		shared_pos [threadIdx.x] = prefetcher;	// Copy prefetched element into shared memory.
		prefetcher = glm::vec4 (0);
		__syncthreads();		

		if (index < N)
			if (j < floor (N / (float)blockDim.x))		// Prefetching to stop at the penultimate block, after the final
				prefetcher = their_pos [index];			// block has been loaded.

		for (int i = 0; i < blockDim.x; i ++)
		{	
			if (j == (floor (N / (float)blockDim.x)))
				if (i >= (N%blockDim.x))
					break;

			if (isApproximately (shared_pos [i].x, my_pos.x) && 
				isApproximately (shared_pos [i].y, my_pos.y) && 
				isApproximately (shared_pos [i].z, my_pos.z))
				continue;

			acc += calculateAcceleration(my_pos, shared_pos [i]);
		}
	}

	// Calculate acceleration due to star.
	acc += calculateAcceleration (my_pos, glm::vec4 (0, 0, 0, starMass));
	
	return acc;
}

// Calculate flocking velocity.
// Written by Rohith Chandran.
__device__ glm::vec3 FlockGlobal (int N, float DT, glm::vec4 my_pos, glm::vec4 *pos, glm::vec3 *vel)
{
	glm::vec3	acc = glm::vec3 (0);
	glm::vec3 my_vel;

	glm::vec3 sumVelocities = glm::vec3 (0);
	glm::vec3 sumPositions = glm::vec3 (0);
	glm::vec3 sumSepVelocities = glm::vec3 (0);

	int neighbours = 0;

	int index = threadIdx.x + (blockIdx.x * blockDim.x);

	if (index < N)
	{
		my_vel = vel [index];

		for (int i = 0; i < N; i ++)
		{	
			glm::vec4 curPos = pos [i];
			float distance = glm::length (curPos - my_pos);
			
			if (distance <= 5.0)
			{
				sumVelocities += vel [i];

				sumPositions.x += curPos.x;
				sumPositions.y += curPos.y;
				sumPositions.z += curPos.z;

				sumSepVelocities.x += (my_pos.x - curPos.x);
				sumSepVelocities.y += (my_pos.y - curPos.y);
				sumSepVelocities.z += (my_pos.z - curPos.z);

				neighbours ++;
			}
		}

		if (neighbours > 0)
		{
			sumSepVelocities /= neighbours;
			sumPositions /= neighbours;		// Centre of mass.
			sumVelocities /= neighbours;	
		}

		// Calculate total velocity:
		glm::vec3 flockVel = (safeNormalize (sumVelocities) * glm::length (my_vel))//	* 0.4f							// Align component
							 +	(safeNormalize (sumPositions - glm::vec3 (my_pos)) * glm::length (my_vel))//	* 0.2f	// Cohesion component
							 +	(safeNormalize (sumSepVelocities) * glm::length (my_vel));// * 0.4f;					// Separation component

		acc = ((glm::length (flockVel) - glm::length (my_vel))/DT) * safeNormalize (flockVel);
	}
	return acc;
}

// normalize only if length > 0
// Written by Rohith Chandran.
inline __device__ glm::vec3 safeNormalize (glm::vec3 vectorToBeNormalized)
{
	float len = glm::length (vectorToBeNormalized);
	if (len > 0.01)
		return vectorToBeNormalized / len;
	return vectorToBeNormalized;
}

// Calculate acceleration for Custom Simulation (flocking).
// Written by Rohith Chandran.
__global__
void updateFCustom (int N, float dt, glm::vec4 * pos, glm::vec3 * vel, glm::vec3 * acc)
{
    int index = threadIdx.x + (blockIdx.x * blockDim.x);
    glm::vec4 my_pos;
    glm::vec3 accel;

    if(index < N) my_pos = pos[index];

    accel = FLOCK (N, dt, my_pos, pos, vel) + calculateAcceleration (my_pos, glm::vec4 (0, 0, 0, starMass));

    if(index < N) acc[index] = accel;
}

//	Update state using Verlet Integration
//	Written by Rohith Chandran.
__global__
void updateS_V (int N, float dt, glm::vec4 * pos, glm::vec3 * vel, glm::vec3 * acc, glm::vec4 * dev_campos, bool cameraUpdate)
{
    int index = threadIdx.x + (blockIdx.x * blockDim.x);
    if( index < N )
    {
		bool isCameraAttached = false;
		glm::vec4	curPos = pos [index];
		glm::vec3	curVel = vel [index];
		glm::vec4	prevPos = curPos - (dt * glm::vec4 (curVel.x, curVel.y, curVel.z, 0));
		
		if (cameraUpdate)
		{
			if (isApproximately (curPos.x, dev_campos->x) &&
				isApproximately (curPos.y, dev_campos->y+CAMFORWARD) &&
				isApproximately (curPos.z, dev_campos->z-CAMHEIGHT))
				isCameraAttached = true;
		}

        curPos = (2.0*curPos) - prevPos + (dt * dt * glm::vec4 (acc [index].x, acc [index].y, acc [index].z, 0));
		prevPos = (curPos - prevPos) / (2.0*dt);
		curVel.x = prevPos.x;
		curVel.y = prevPos.y;
		curVel.z = prevPos.z;
		
		vel[index] = curVel;
        pos[index] = curPos;
		
		if (isCameraAttached)
		{
			dev_campos->z = curPos.z + CAMHEIGHT;
			dev_campos->y = curPos.y - CAMFORWARD;
		}
    }
}

//	Update state using Leapfrog Integration.
//	Written by Rohith Chandran.
__global__
void updateS_LF (int N, float dt, glm::vec4 * pos, glm::vec3 * vel, glm::vec3 * acc, glm::vec4 * dev_campos, bool cameraUpdate)
{
    int index = threadIdx.x + (blockIdx.x * blockDim.x);
    if( index < N )
    {
        /*bool isCameraAttached = false;*/
		glm::vec4 curPos = pos[index];
		/*if (cameraUpdate)
		{
			if (isApproximately (curPos.x, dev_campos->x) &&
				isApproximately (curPos.y, dev_campos->y+CAMFORWARD) &&
				isApproximately (curPos.z, dev_campos->z-CAMHEIGHT))
				isCameraAttached = true;
		}*/
		
		curPos += glm::vec4 ((vel [index] * dt), 0);
		vel[index] += acc [index] * dt;

		if (/*isCameraAttached*/cameraUpdate)
		{
			if (index == attachedToIndex)
			{
				dev_campos->z = curPos.z + CAMHEIGHT;
				dev_campos->y = curPos.y - CAMFORWARD;
			}
		}
		pos [index] = curPos;
    }
}

//	Sets up velocity for leapfrog integration.
//	Written by Rohith Chandran.
__global__
void setupVelocityLF (int N, float dt, glm::vec3 * vel, glm::vec3 * acc)
{
    int index = threadIdx.x + (blockIdx.x * blockDim.x);
    if( index < N )
		vel[index] += acc [index] * dt;
}

__global__	void	moveCamera (int N, glm::vec4* campos, glm::vec4* pos)
{
	for (int i = 0;	i < N; i ++)
	{
		if (i <= attachedToIndex)
		{
			/*if (attachedToIndex == N)
				attachedToIndex = 0;
			else*/
				continue;
		}

		if (glm::length (pos [i] - pos [attachedToIndex]) > 5.0)
		{
			*campos = pos [i];
			campos->z += CAMHEIGHT;
			campos->y -= CAMFORWARD;

			attachedToIndex = i;

			break;
		}
	}
}

__global__ void attachedToIndexReset ()
{
	attachedToIndex = 0;
}

/*************************************
 * Wrappers for the __global__ calls *
 *************************************/

//Initialize memory, update some globals
void initCuda(int N, const glm::vec4 &camera_position)
{
    numObjects = N;
    dim3 fullBlocksPerGrid((int)ceil(float(N)/float(blockSize)));

    cudaMalloc((void**)&dev_pos, N*sizeof(glm::vec4));
    checkCUDAErrorWithLine("Kernel failed!");
    cudaMalloc((void**)&dev_vel, N*sizeof(glm::vec3));
    checkCUDAErrorWithLine("Kernel failed!");
    cudaMalloc((void**)&dev_acc, N*sizeof(glm::vec3));
    checkCUDAErrorWithLine("Kernel failed!");
	cudaMalloc((void**)&dev_campos, sizeof(glm::vec4));
    checkCUDAErrorWithLine("Kernel failed!");

	resetAttachedToIndex ();
	checkCUDAErrorWithLine("Kernel failed!");
	cudaMemcpy (dev_campos, &camera_position, sizeof (camera_position), cudaMemcpyHostToDevice);
    generateRandomPosArray<<<fullBlocksPerGrid, blockSize>>>(1, numObjects, dev_pos, scene_scale, planetMass);
    checkCUDAErrorWithLine("Kernel failed!");
    generateCircularVelArray<<<fullBlocksPerGrid, blockSize>>>(2, numObjects, dev_vel, dev_pos);
    checkCUDAErrorWithLine("Kernel failed!");
    cudaThreadSynchronize();

	float dt = 0.1;
	updateF<<<fullBlocksPerGrid, blockSize, blockSize*sizeof(glm::vec4)>>>(numObjects, dt, dev_pos, dev_vel, dev_acc);
	setupVelocityLF<<<fullBlocksPerGrid, blockSize>>>(numObjects, dt, dev_vel, dev_acc);
}

void cudaNBodyUpdateWrapper(float dt, bool customSimulation)
{
    dim3 fullBlocksPerGrid((int)ceil(float(numObjects)/float(blockSize)));
	if (customSimulation)
		updateFCustom<<<fullBlocksPerGrid, blockSize/*, blockSize*sizeof(glm::vec4)*/>>>(numObjects, dt, dev_pos, dev_vel, dev_acc);
	else
		updateF<<<fullBlocksPerGrid, blockSize, blockSize*sizeof(glm::vec4)>>>(numObjects, dt, dev_pos, dev_vel, dev_acc);
    checkCUDAErrorWithLine("Kernel failed!");

//	updateS<<<fullBlocksPerGrid, blockSize>>>(numObjects, dt, dev_pos, dev_vel, dev_acc, dev_campos, camUpdate);
//  updateS_V<<<fullBlocksPerGrid, blockSize>>>(numObjects, dt, dev_pos, dev_vel, dev_acc, dev_campos, camUpdate);
	updateS_LF<<<fullBlocksPerGrid, blockSize>>>(numObjects, dt, dev_pos, dev_vel, dev_acc, dev_campos, camUpdate);
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

void setDevicePrefetch (bool prefetchEnabled)
{
	cudaMemcpyToSymbol (&prefetch, &prefetchEnabled, sizeof (bool), 0);
}

glm::vec4	getCurrentCameraPosition ()
{
	glm::vec4	camera_position;
	cudaMemcpy (&camera_position, dev_campos, sizeof (camera_position), cudaMemcpyDeviceToHost);
	return camera_position/scene_scale;
}

glm::vec3	getCurrentCameraLookAt ()
{
	int attachedTo = 0;
	cudaMemcpy (&attachedTo, &attachedToIndex, sizeof (int), cudaMemcpyDeviceToHost);
	checkCUDAErrorWithLine("Kernel failed!");
	glm::vec3	accelerationDir = glm::vec3 (0);
	glm::vec3	pos = glm::vec3 (0);
	cudaMemcpy (&accelerationDir, &dev_acc [attachedTo], sizeof (glm::vec3), cudaMemcpyDeviceToHost);
	checkCUDAErrorWithLine("Kernel failed!");
	cudaMemcpy (&pos, &dev_pos [attachedTo], sizeof (glm::vec3), cudaMemcpyDeviceToHost);
	checkCUDAErrorWithLine("Kernel failed!");
	pos += accelerationDir * 5.0f;
	return	pos/scene_scale;
}

void		setCurrentCameraPosition (const glm::vec4 &camera_position)
{
	glm::vec4	cPos = camera_position * scene_scale;
	cudaMemcpy (dev_campos, &cPos, sizeof (glm::vec4), cudaMemcpyHostToDevice);
}

void	moveCameraToNextFlock (glm::vec3 &cameraPos)
{
	moveCamera<<<1,1>>> (numObjects, dev_campos, dev_pos);
}

void	setCameraUpdate (bool shouldCameraUpdate)
{
	camUpdate = shouldCameraUpdate;
}

void	resetAttachedToIndex ()
{
	attachedToIndexReset<<<1,1>>> ();
}