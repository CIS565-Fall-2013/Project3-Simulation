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
const float planetMass = MASS;
const __device__ float starMass = 5e10;


//size of the height map in simulation space

glm::vec4 * dev_pos;
glm::vec3 * dev_vel;
glm::vec3 * dev_acc;

float * dev_density;
float * dev_pressure;

glm::vec3 * dev_force;

// Extra arrays for more advanced integration techniques
glm::vec3 * dev_acc_2;

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

/********************** SPH Functions **********************/

// kernels from 
// http://www.matthiasmueller.info/publications/sca03.pdf
// More reference for understanding from
// http://andrew.gibiansky.com/blog/physics/computational-fluid-dynamics/

__device__ float kernel_general(float xij)
{

	if ( xij > kernelSize)
		return 0.0f;
	else
	// poly6 kernel
		return (315.0 * (kernelSizeSqr - xij*xij) * (kernelSizeSqr - xij*xij) * (kernelSizeSqr - xij*xij))/
				(64.0 * pi * kernelSizeSqr * kernelSizeSqr * kernelSizeSqr * kernelSizeSqr * kernelSize);
	/*
	// spiky kernel
		return (15.0 * (kernelSize - xij) * (kernelSize - xij) * (kernelSize - xij)) / 
				( pi * kernelSizeSqr * kernelSizeSqr * kernelSizeSqr);
	// viscous kernel
		return (15.0 * (-(0.5*(xij*xij*xij)/(kernelSizeSqr * kernelSize)) + ((xij*xij)/kernelSizeSqr) + (0.5 * kernelSize/xij) - 1.0) ) /
				(2.0 * pi * kernelSizeSqr * kernelSize);
	*/

}

// Check general grad and lap
__device__ float kernel_general_gradient(float xij)
{
	if ( xij > kernelSize)
		return 0.0f;
	else
		return (-945 * xij * (kernelSizeSqr - xij*xij) * (kernelSizeSqr - xij*xij))/
				(32 * pi * kernelSizeSqr * kernelSizeSqr * kernelSizeSqr * kernelSizeSqr * kernelSize);
}

__device__ float kernel_general_laplacian(float xij)
{
	if ( xij > kernelSize)
		return 0.0f;
	else
		return (945 * (kernelSizeSqr - xij*xij) * (4*xij*xij  - (kernelSizeSqr - xij*xij) ) )/
				(32 * pi * kernelSizeSqr * kernelSizeSqr * kernelSizeSqr * kernelSizeSqr * kernelSize);
}

__device__ float kernel_visc_laplacian(float xij)
{

	if ( xij > kernelSize)
		return 0.0f;
	else

	// viscous kernel
	/*		(15.0 * (-(0.5*(xij*xij*xij)/(kernelSizeSqr * kernelSize)) + ((xij*xij)/kernelSizeSqr) + (0.5 * kernelSize/xij) - 1.0) ) /
				(2.0 * pi * kernelSizeSqr * kernelSize);
	*/
		return (45.0 * (kernelSize - xij))/(pi * kernelSizeSqr * kernelSizeSqr * kernelSizeSqr);

}

__device__ float kernel_press_gradient(float xij)
{
	if ( xij > kernelSize)
		return 0.0f;
	else

	// viscous kernel
	/*		(15.0 * (-(0.5*(xij*xij*xij)/(kernelSizeSqr * kernelSize)) + ((xij*xij)/kernelSizeSqr) + (0.5 * kernelSize/xij) - 1.0) ) /
				(2.0 * pi * kernelSizeSqr * kernelSize);
	*/
		return (- 45.0 * (kernelSize - xij) * (kernelSize - xij))/(pi * kernelSizeSqr * kernelSizeSqr * kernelSizeSqr);

}

/***********************************************************/


//Generate randomized starting positions for the planets in the XY plane
//Also initialized the masses
__global__
void generateRandomPosArray(int time, int N, glm::vec4 * arr, float scale, float mass)
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if(index < N)
    {
		// go from -xmax to xmax, -ymax to ymax and z = 0 upwards

		glm::vec3 llcorner = glm::vec3(-B_XMAX,-B_YMAX,B_ZMIN);
		int numberOfSpheresInX = (2*B_XMAX) / (2*RADIUS);
		int numberOfSpheresInY = (2*B_YMAX) / (2*RADIUS);

		int z = index * 1.0f / (numberOfSpheresInX * numberOfSpheresInY);
		int y = (index - (z * numberOfSpheresInX * numberOfSpheresInY)) / numberOfSpheresInX;
		int x = index - (z * numberOfSpheresInX * numberOfSpheresInY) - (y * numberOfSpheresInX);

		arr[index].x = llcorner.x + 2*RADIUS*x;
		arr[index].y = llcorner.y + 2*RADIUS*y;
		arr[index].z = llcorner.z + 2*RADIUS*z;
		/*
        glm::vec3 rand = scale*(generateRandomNumberFromThread(time, index)-0.5f);
        arr[index].x = rand.x;
        arr[index].y = rand.y;
        arr[index].z = 0.45f * (rand.z + 0.5f * scale);
		*/
        arr[index].w = mass;
    }
}

//Determine velocity from the distance from the center star. Not super physically accurate because 
//the mass ratio is too close, but it makes for an interesting looking scene
__global__ void generateCircularVelArray(int time, int N, glm::vec3 * arr, glm::vec4 * pos)
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if(index < N)
    {
        //glm::vec3 R = glm::vec3(pos[index].x, pos[index].y, pos[index].z);
        //float r = glm::length(R) + EPSILON;
        //float s = sqrt(G*starMass/r);
        //glm::vec3 D = glm::normalize(glm::cross(R/r,glm::vec3(0,0,1)));
        arr[index].x = 0;//s*D.x;
        arr[index].y = 0;//s*D.y;
        arr[index].z = 0;//s*D.z;
    }
}

//Generate randomized starting velocities in the XY plane
__global__ void generateRandomVelArray(int time, int N, glm::vec3 * arr, float scale)
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if(index < N)
    {
        glm::vec3 rand = scale*(generateRandomNumberFromThread(time, index) - 0.5f);
        arr[index].x = 0;//rand.x;
        arr[index].y = 0;//rand.y;
        arr[index].z = 0;//rand.z;
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
	
	glm::vec3 myPos = glm::vec3(  us.x,  us.y,  us.z);
	glm::vec3 urPos = glm::vec3(them.x,them.y,them.z);
	glm::vec3 direction = urPos - myPos;
    float distance = glm::length(direction);
	
	if(distance > 0.000001f)
	{
		float GConst = G;
		return GConst * them.w * (1.0f/glm::dot(direction,direction)) * glm::normalize(direction);
	}

	else
		return glm::vec3(0.0f);
}

//TODO: Core force calc kernel global memory
__device__ 
glm::vec3 naiveAcc(int N, glm::vec4 my_pos, glm::vec4 * their_pos)
{	return GRAVITY;

	glm::vec3 acc = calculateAcceleration(my_pos, glm::vec4(0,0,0,starMass));
	
	for(int i=0; i< N ; i++)
	{
		acc += calculateAcceleration(my_pos, their_pos[i]);
	}
    return acc;
}


//TODO: Core force calc kernel shared memory
__device__ glm::vec3 sharedMemAcc(int N, glm::vec4 my_pos, glm::vec4 * their_pos)
{
	return GRAVITY;

	__shared__ glm::vec4 sharedBodyData[blockSize];
	int tid = threadIdx.x;
	int bid = blockIdx.x;

    glm::vec3 acc = glm::vec3(0.0f); //calculateAcceleration(my_pos, glm::vec4(0,0,0,starMass));
	int numberOfLoops = ceil((1.0f*N) / blockSize);

	for(int i=0; i<numberOfLoops; i++)
	{
		int deltaIndex = ((i + bid)%numberOfLoops );
		int index = deltaIndex * blockSize + tid;
		if( index < N)
		{
			sharedBodyData[tid] = their_pos[index];
		}
		__syncthreads();
		for(int j=0; j< blockSize; j++)
		{
			if((deltaIndex * blockSize + j < N) )
				acc += calculateAcceleration(my_pos,sharedBodyData[j]);
		}
	}

    return acc;
}

__global__ void calculateSPHDensityPressure(int N, glm::vec4 * pos, float * density, float * pressure)
{
	__shared__ glm::vec4 sharedPos[blockSize];
    int index = threadIdx.x + (blockIdx.x * blockDim.x);
	int tid = threadIdx.x;
	int bid = blockIdx.x;
	int numberOfLoops = ceil((1.0f*N) / blockSize);
	
	glm::vec4 myPos;

	if(index < N)
		myPos = pos[index];

	float myDensity = 0;
	for(int i=0; i<numberOfLoops; i++)
	{
		// For this block'd data, bring it into shared  memory
		int deltaIndex = ((i + bid)%numberOfLoops );
		int delIndex = deltaIndex * blockSize + tid;
		if(delIndex < N)
			sharedPos[tid] = pos[delIndex];
		__syncthreads();

		// Traverse shared memory
		for(int j=0; j< blockSize; j++)
		{
			if( index < N && (deltaIndex * blockSize + j < N))
			{
				glm::vec4 theirPos = sharedPos[j];
				glm::vec3 r = glm::vec3(myPos.x-theirPos.x, myPos.y-theirPos.y, myPos.z-theirPos.z);
				float xij = glm::length(r);
				float W = kernel_general(xij);
				myDensity += theirPos.w * W;
			}
		}
		__syncthreads();
	}
	
	if(index < N)
	{
		density[index] = myDensity;	
		pressure[index] = STIFFNESS * (myDensity - REF_DENSITY);
	}
}

__global__ void calculateSPHForces(int N, glm::vec4 * pos, glm::vec3 * vel, float *density, float * pressure, glm::vec3 * acc)
{
	__shared__ glm::vec4 sharedPositions[blockSize];
	__shared__ float sharedDensity[blockSize];
	__shared__ float sharedPressure[blockSize];
	__shared__ glm::vec3 sharedVelocity[blockSize];

    int index = threadIdx.x + (blockIdx.x * blockDim.x);
	int tid = threadIdx.x;
	int bid = blockIdx.x;
	int numberOfLoops = ceil((1.0f*N) / blockSize);
	
	glm::vec4 myPos;
	float myDens;
	float myPress;
	glm::vec3 myVel;

	if(index < N)
	{
		myPos = pos[index];
		myDens = density[index];
		myPress = pressure[index];
		myVel = vel[index];
	}
	glm::vec3 pressureAcc = glm::vec3(0);
	glm::vec3 viscosityAcc = glm::vec3(0);
	glm::vec3 surfaceAcc = glm::vec3(0);

	for(int i=0; i<numberOfLoops; i++)
	{
		// For this block'd data, bring it into shared  memory
		int deltaIndex = ((i + bid)%numberOfLoops );
		int delIndex = deltaIndex * blockSize + tid;
		if(delIndex < N)
		{
			sharedPositions[tid] = pos[delIndex];
			sharedDensity[tid] = density[delIndex];
			sharedPressure[tid] = pressure[delIndex];
			sharedVelocity[tid] = vel[delIndex];
		}
		__syncthreads();

		// Traverse shared memory
		for(int j=0; j< blockSize; j++)
		{
			if( index < N && (deltaIndex * blockSize + j < N))
			{
				glm::vec4 theirPos = sharedPositions[j];
				float theirDens = sharedDensity[j];
				float theirPress = sharedPressure[j];
				glm::vec3 theirVel = sharedVelocity[j];
				glm::vec3 r = glm::vec3(myPos.x-theirPos.x, myPos.y-theirPos.y, myPos.z-theirPos.z);
				float xij = glm::length(r);
				
				// Symmetrization based on Monaghan
				float pressureTerm = - theirPos.w * (myPress/(myDens*myDens) + theirPress/(theirDens*theirDens) ) * kernel_press_gradient(xij);
				if(pressureTerm == pressureTerm && fabs(myDens) > EPSIL && fabs(theirDens) > EPSIL) // NaN check
					pressureAcc += pressureTerm * r/(xij + EPSIL);
				
				if(myDens==myDens && theirDens == theirDens && fabs(myDens) > EPSIL && fabs(theirDens) > EPSIL) // NaN check
					viscosityAcc += VISCOSITY * theirPos.w * (theirVel - myVel)/(myDens * theirDens) * kernel_visc_laplacian(xij);
				
			}
		}
		__syncthreads();
	}
	
	if(index < N)
	{
		acc[index] = pressureAcc + viscosityAcc + surfaceAcc +  GRAVITY;

		//printf("%3d: %3.3f %3.3f %3.3f\n",index, acc[index].x, acc[index].y, acc[index].z);
		//printf("\tp: %3.3f %3.3f %3.3f \n", pressureAcc.x, pressureAcc.y, pressureAcc.z);
		//printf("\tv: %3.3f %3.3f %3.3f \n", viscosityAcc.x, viscosityAcc.y, viscosityAcc.z);
	}
}

__global__ void updateSPHExplicit(int N, float dt, glm::vec4 * pos, glm::vec3 * vel, glm::vec3 * acc)
{
	int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if(index < N)
	{
		glm::vec3 newVel = vel[index] + dt * acc[index];
		vel[index] = newVel;
		pos[index].x += dt * newVel.x;
		pos[index].y += dt * newVel.y;
		pos[index].z += dt * newVel.z;
	}
}

//Simple Euler integration scheme
__global__ void updateF(int N, float dt, glm::vec4 * pos, glm::vec3 * vel, glm::vec3 * acc)
{
    int index = threadIdx.x + (blockIdx.x * blockDim.x);
	glm::vec4 my_pos = glm::vec4(0,0,0,1);
    if( index < N )
       my_pos = pos[index];
	glm::vec3 accel = ACC(N, my_pos, pos);
	if( index < N )
		acc[index] = accel;
}
__global__ void updateP(int N, float dt, glm::vec4 * pos, glm::vec3 * vel, glm::vec3 * acc)
{
    int index = threadIdx.x + (blockIdx.x * blockDim.x);
    if( index < N )
	{
        vel[index] += acc[index] * dt;
        pos[index].x += vel[index].x * dt;
        pos[index].y += vel[index].y * dt;
        pos[index].z += vel[index].z * dt;
    }
}

__global__ void updateVelVerletPart1F(int N, float dt, glm::vec4 * pos, glm::vec3 * vel, glm::vec3 * acc)
{
    int index = threadIdx.x + (blockIdx.x * blockDim.x);
	glm::vec4 my_pos;
	glm::vec3 accel;
    if( index < N )
		my_pos = pos[index];
	accel = ACC(N,my_pos,pos);
	if( index < N )
		acc[index] = accel;
}
__global__ void updateVelVerletPart1P(int N, float dt, glm::vec4 * pos, glm::vec3 * vel, glm::vec3 * acc)
{
    int index = threadIdx.x + (blockIdx.x * blockDim.x);

	if( index < N )
	{
		glm::vec3 deltaPos = dt * (vel[index] + dt * 0.5f * acc[index]);
		pos[index].x += deltaPos.x;
		pos[index].y += deltaPos.y;
		pos[index].z += deltaPos.z;
	}
}
__global__ void updateVelVerletPart2F(int N, float dt, glm::vec4 * pos, glm::vec3 * vel, glm::vec3 * acc_2)
{
    int index = threadIdx.x + (blockIdx.x * blockDim.x);
	glm::vec4 my_pos;
	glm::vec3 accel;
    if( index < N )
		my_pos = pos[index];
	accel = ACC(N,my_pos,pos);
	if( index < N )
		acc_2[index] = accel;
}

__global__ void updateVelVerletPart2P(int N, float dt, glm::vec4 * pos, glm::vec3 * vel, glm::vec3 * acc, glm::vec3  * acc_2)
{
    int index = threadIdx.x + (blockIdx.x * blockDim.x);
    if( index < N )
    {
		vel[index] += dt * 0.5f * (acc[index] + acc_2[index]);
	}
}

__global__ void handleCollisions(int N, glm::vec4 * pos, glm::vec3 * vel)
{
    int index = threadIdx.x + (blockIdx.x * blockDim.x);
    if( index < N )
    {
		glm::vec4 myPos = pos[index];
		glm::vec3 myVel = vel[index];

		// Simple impulse collision handling
		// Down side, adds energy into the system
		// Consider stiff springs with collision buffers
		if( myPos.z < 0.0f)
		{
			myPos.z = EPSIL;
			myVel.z = -DRAG*myVel.z;
		}
		if( myPos.z > ZMAX)
		{
			myPos.z = ZMAX - EPSIL;
			myVel.z = -DRAG*myVel.z;
		}
		if(myPos.y < -YMAX)
		{
			myPos.y = -(YMAX - EPSIL);
			myVel.y = -DRAG*myVel.y;
		}
		if(myPos.y > YMAX)
		{
			myPos.y = YMAX - EPSIL;
			myVel.y = -DRAG*myVel.y;
		}
		if(myPos.x < -XMAX)
		{
			myPos.x = -(XMAX - EPSIL);
			myVel.x = -DRAG*myVel.x;
		}
		if(myPos.x > XMAX)
		{
			myPos.x = XMAX - EPSIL;
			myVel.x = -DRAG*myVel.x;
		}
		pos[index] = myPos;
		vel[index] = myVel;
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
	// z is negative in downward direction
	float c_scale_z = 2.0f / s_scale;

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

    cudaMalloc((void**)&dev_pos, N*sizeof(glm::vec4));
    checkCUDAErrorWithLine("Kernel failed!");
    cudaMalloc((void**)&dev_vel, N*sizeof(glm::vec3));
    checkCUDAErrorWithLine("Kernel failed!");
	cudaMalloc((void**)&dev_acc, N*sizeof(glm::vec3));
	checkCUDAErrorWithLine("Kernel failed!");

	// SPH
	cudaMalloc((void**)&dev_density, N*sizeof(float));
	checkCUDAErrorWithLine("Kernel failed!");
	cudaMalloc((void**)&dev_pressure, N*sizeof(float));
	checkCUDAErrorWithLine("Kernel failed!");
	cudaMalloc((void**)&dev_force, N*sizeof(glm::vec3));
	checkCUDAErrorWithLine("Kernel failed!");


	// For velocityVerlet
	cudaMalloc((void**)&dev_acc_2, N*sizeof(glm::vec3));
	checkCUDAErrorWithLine("Kernel failed!");

    generateRandomPosArray<<<fullBlocksPerGrid, blockSize>>>(1, numObjects, dev_pos, scene_scale, planetMass);
    checkCUDAErrorWithLine("Kernel failed!");
    generateCircularVelArray<<<fullBlocksPerGrid, blockSize>>>(2, numObjects, dev_vel, dev_pos);
    checkCUDAErrorWithLine("Kernel failed!");
}

void cudaCollisionsWrapper()
{
	dim3 fullBlocksPerGrid((int)ceil(float(numObjects)/float(blockSize)));
	handleCollisions<<<fullBlocksPerGrid, blockSize>>>(numObjects, dev_pos, dev_vel);
	checkCUDAErrorWithLine("Kernel failed!");
}

void cudaNBodyUpdateWrapper(float dt)
{
    dim3 fullBlocksPerGrid((int)ceil(float(numObjects)/float(blockSize)));
    updateF<<<fullBlocksPerGrid, blockSize>>>(numObjects, dt, dev_pos, dev_vel, dev_acc);
	updateP<<<fullBlocksPerGrid, blockSize>>>(numObjects, dt, dev_pos, dev_vel, dev_acc);
    checkCUDAErrorWithLine("Kernel failed!");

	cudaCollisionsWrapper();
}

void cudaNBodyUpdateVelocityVerletWrapper(float dt)
{
	dim3 fullBlocksPerGrid((int)ceil(float(numObjects)/float(blockSize)));
    updateVelVerletPart1F<<<fullBlocksPerGrid, blockSize>>>(numObjects, dt, dev_pos, dev_vel, dev_acc);
    checkCUDAErrorWithLine("Kernel failed!");
    updateVelVerletPart1P<<<fullBlocksPerGrid, blockSize>>>(numObjects, dt, dev_pos, dev_vel, dev_acc);
    checkCUDAErrorWithLine("Kernel failed!");


    updateVelVerletPart2F<<<fullBlocksPerGrid, blockSize>>>(numObjects, dt, dev_pos, dev_vel, dev_acc_2);
    checkCUDAErrorWithLine("Kernel failed!");
    updateVelVerletPart2P<<<fullBlocksPerGrid, blockSize>>>(numObjects, dt, dev_pos, dev_vel, dev_acc, dev_acc_2);
    checkCUDAErrorWithLine("Kernel failed!");

	cudaCollisionsWrapper();
}

void cudaSPHUpdateWrapper(float dt)
{
	dim3 fullBlocksPerGrid((int)ceil(float(numObjects)/float(blockSize)));
	// Calculate Density and Pressures
	calculateSPHDensityPressure<<<fullBlocksPerGrid, blockSize>>>(numObjects,dev_pos,dev_density,dev_pressure);
	checkCUDAErrorWithLine("Kernel failed!");
	// Calculate Accelerations
	calculateSPHForces<<<fullBlocksPerGrid, blockSize>>>(numObjects,dev_pos,dev_vel,dev_density,dev_pressure,dev_acc);
	checkCUDAErrorWithLine("Kernel failed!");
	// Update Positions and Velocities
	updateSPHExplicit<<<fullBlocksPerGrid, blockSize>>>(numObjects,dt,dev_pos,dev_vel,dev_acc);
	checkCUDAErrorWithLine("Kernel failed!");
	// Check for collisions with boundaries
	cudaCollisionsWrapper();
	checkCUDAErrorWithLine("Kernel failed!");


}


void cudaUpdateVBO(float * vbodptr, int width, int height)
{
    dim3 fullBlocksPerGrid((int)ceil(float(numObjects)/float(blockSize)));
    sendToVBO<<<fullBlocksPerGrid, blockSize>>>(numObjects, dev_pos, vbodptr, width, height, scene_scale);
    checkCUDAErrorWithLine("Kernel failed!");
}

void cudaUpdatePBO(float4 * pbodptr, int width, int height)
{
    dim3 fullBlocksPerGrid((int)ceil(float(width*height)/float(blockSize)));
    sendToPBO<<<fullBlocksPerGrid, blockSize>>>(numObjects, dev_pos, pbodptr, width, height, scene_scale);
    checkCUDAErrorWithLine("Kernel failed!");
}
