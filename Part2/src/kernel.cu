#include <stdio.h>
#include <cstdio>
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
const __device__ float planetMass = 3e8;
const __device__ float starMass = 6e10;

const float scene_scale = 2e2; //size of the height map in simulation space

const __device__ float vel_mag = 10.0f; // the magnitude of velocity
__device__ float neighbor[3] = {100,80,50};

glm::vec4 * dev_pos;
glm::vec3 * dev_vel;
glm::vec3 * dev_acc;
glm::vec4 * poscopy;// used for thrust
s_Steer * stePointer;
s_Steer steer;
//float * steeringForce; // the force that make birds come back from outside
//glm::vec3 * targetPos; // targetPos that steer birds to, generated each time in update function

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
float length(glm::vec3 vec)
{
	return sqrt(vec.x*vec.x+vec.y*vec.y+vec.z*vec.z);
}
__host__ __device__
glm::vec3 normalize(glm::vec3 vec)
{
	if(length(vec)<EPSILON) return glm::vec3(0,0,0);
	return vec/length(vec);
}
__host__ __device__
glm::vec4 normalize(glm::vec4 vec)
{
	glm::vec3 v(vec.x,vec.y,vec.z);
	if(length(v) < EPSILON) return glm::vec4(0,0,0,0);
	return glm::vec4(v/length(v),vec.w);
}
__host__ __device__
float dot(glm::vec3 v1,glm::vec3 v2)
{
	return v1.x*v2.x + v1.y*v2.y + v1.z*v2.z;
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
__host__ __device__
void printvec(glm::vec3 a)
{
	printf("%f,%f,%f   ",a.x,a.y,a.z);
}
__host__ __device__
void printfloat(float a)
{
	printf("float: %f   |",a);
}

//Function that generates static.
__host__ __device__ 
glm::vec3 generateRandomNumberFromThread(float time, int index)
{
    thrust::default_random_engine rng(hash(index*time));
    thrust::uniform_real_distribution<float> u01(0,1);

    return glm::vec3((float) u01(rng), (float) u01(rng), (float) u01(rng));
}
__host__ __device__
float generateRandomFloat(float min,float max,float time,int index)
{
	thrust::default_random_engine rng(hash(index*time));
    thrust::uniform_real_distribution<float> u01(min,max);
	return (float)u01(rng);
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
#if PART2 == 1
        arr[index].z = rand.z;//.0f;//rand.z;
#else
		arr[index].z = 0.0;
#endif
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
        float r = length(R) + EPSILON;
        float s = sqrt(G*starMass/r);
        glm::vec3 D = normalize(glm::cross(R/r,glm::vec3(0,0,1)));
        arr[index].x = s*D.x;
        arr[index].y = s*D.y;
        arr[index].z = s*D.z;
    }
}

//Generate randomized starting velocities in the XY plane
__global__
void generateRandomVelArray(int time, int N, glm::vec3 * arr, float scale, float velMag)
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if(index < N)
    {
        glm::vec3 rand = scale*(generateRandomNumberFromThread(time, index) - 0.5f);
#if PART2 == 1
		rand = normalize(rand);		
        arr[index].x = rand.x * velMag;
        arr[index].y = rand.y * velMag;
        arr[index].z = rand.z * velMag;//0.0;//
#else
		arr[index].x = rand.x;
        arr[index].y = rand.y;
        arr[index].z = 0.0;//rand.z;
#endif

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

	glm::vec3 us_pos(us.x,us.y,us.z);
	glm::vec3 them_pos(them.x,them.y,them.z);
	glm::vec3 dir = them_pos - us_pos;
	float rsquare = length(dir);
	if(rsquare < EPSILON) return glm::vec3(0);
	rsquare *= rsquare;
	float aMag  = G * them.w /rsquare;
	glm::vec3 a = aMag * normalize(dir);
	return a;
}
__host__ __device__
bool isInRange(glm::vec4 us,glm::vec4 them,glm::vec3 my_vel,int type)
//type is for different forces
//0: flocking, 1:cohesion, 2:seperation
{
	//tell if distance is within the neighbor range
	glm::vec3 myPos(us.x,us.y,us.z);
	glm::vec3 theirPos(them.x,them.y,them.z);
	glm::vec3 dir = theirPos - myPos;
	float dist = length(dir);
	/*if(dist < EPSILON) 
	`	return false;*/
	if(dist > neighbor[type])
		return false;
	//tell if direction is within the eye range
	dir = normalize(dir);
	my_vel = normalize(my_vel);
	float angle = acosf(dot(dir,my_vel));
	if(abs(angle) > VIEWANGLE)
		return false;
	return true;
}
__host__ __device__
glm::vec3 flockAcc(int N,glm::vec4 my_pos,glm::vec4 * their_pos,glm::vec3* their_vel,glm::vec3 my_vel,s_Steer steer)
{
	glm::vec3 totalDir;
	glm::vec4 totalPos; // cohesion
	glm::vec4 totalPos2; //separation
	int numInrangeFlock = 0; int numInrangeCo = 0; int numInrangeSep = 0;
	for(int i = 0;i<N;++i)
	{
		if(isInRange(my_pos,their_pos[i],my_vel,0) == true)
		{
			numInrangeFlock ++;
			totalDir += their_vel[i];			
		}
		if(isInRange(my_pos,their_pos[i],my_vel,1) == true)
		{
			numInrangeCo ++;
			totalPos += their_pos[i];
		}
		if(isInRange(my_pos,their_pos[i],my_vel,2) == true)
		{
			numInrangeSep ++;
			totalPos2 += their_pos[i];
		}
		//add steering force
		
	}

	if(numInrangeFlock == 0 && numInrangeCo == 0 && numInrangeSep == 0)
		return glm::vec3(0,0,0);

	glm::vec3 flockAcc = glm::vec3(0);
	glm::vec4 cohesionAcc,separateAcc;
	cohesionAcc = separateAcc = glm::vec4(0);
	
	if(numInrangeFlock>0)
	{		
		glm::vec3 meanDir = totalDir / (float)numInrangeFlock;
		flockAcc = (float)FLOCKFORCE * normalize(meanDir) /planetMass;
	}
	if(numInrangeCo >0)
	{
		//printfloat((float)numInrangeCo);
		glm::vec4 meanPos = totalPos / (float)numInrangeCo;
		cohesionAcc = (float)COHESIONFORCE * normalize(meanPos - my_pos) / planetMass;
		//cohesionAcc = normalize(cohesionAcc) * length(glm::vec3(meanPos.x - my_pos.z, meanPos.y-my_pos.y, meanPos.z-my_pos.z));
	}
	if(numInrangeSep>0)
	{
		//printfloat((float)numInrangeSep);
		glm::vec4 meanPos = totalPos2 / (float)numInrangeSep;
		separateAcc = (float)SEPARATIONFORCE * normalize(my_pos-meanPos) / planetMass;
	}

	glm::vec3 a = flockAcc + glm::vec3(cohesionAcc.x,cohesionAcc.y,cohesionAcc.z) + glm::vec3(separateAcc.x,separateAcc.y,separateAcc.z);

	//glm::vec3 a = 0.5f * normalize(flockAcc) + 0.3f * normalize(glm::vec3(cohesionAcc.x,cohesionAcc.y,cohesionAcc.z)) + 0.2f* normalize(glm::vec3(separateAcc.x,separateAcc.y,separateAcc.z));
	//add steerforce
	glm::vec3 steerAcc = steer.steeringForce * normalize(steer.targetPos - glm::vec3(my_pos.x,my_pos.y,my_pos.z)) / planetMass;
	
	
	if(steer.steeringForce >0)
	{
		//printfloat(steer.steeringForce);
		a = steerAcc + glm::vec3(cohesionAcc.x,cohesionAcc.y,cohesionAcc.z) + glm::vec3(separateAcc.x,separateAcc.y,separateAcc.z);
		//glm::vec3 a = 0.5f * normalize(steerAcc) + 0.3f * normalize(glm::vec3(cohesionAcc.x,cohesionAcc.y,cohesionAcc.z)) + 0.2f* normalize(glm::vec3(separateAcc.x,separateAcc.y,separateAcc.z));
	}
	
	return normalize(a);
}

//TODO: Core force calc kernel global memory
__device__ 
glm::vec3 naiveAcc(int N, glm::vec4 my_pos, glm::vec4 * their_pos)
{	
	glm::vec3 acc = calculateAcceleration(my_pos,glm::vec4(0,0,0,starMass));
	for(int i = 0;i<N;++i)
	{
		acc += calculateAcceleration(my_pos, their_pos[i]);
	}
    return acc;
}


//TODO: Core force calc kernel shared memory
__device__ 
glm::vec3 sharedMemAcc(int N, glm::vec4 my_pos, glm::vec4 * their_pos)
{
    glm::vec3 acc = calculateAcceleration(my_pos, glm::vec4(0,0,0,starMass));	
	__shared__ glm::vec4 sharedPos[blockSize]; // tileSize = blockSize;
	for(int tile = 0, index = 0;index<N;index+= blockSize,tile++)
	{
		int i = 0;
		for(i = 0; i<blockDim.x; i++)
		{
			int index = tile*blockDim.x +i;
			if(index >= N) break;
			sharedPos[i] = their_pos[tile*blockDim.x + i];
		}
		for(int idx = 0;idx<blockDim.x && idx <= i;++idx)
		{
			acc+= calculateAcceleration(my_pos,sharedPos[idx]);
		}
		__syncthreads();
	}
    return acc;
}


//Simple Euler integration scheme
__global__
void updateF(int N, float dt, glm::vec4 * pos, glm::vec3 * vel, glm::vec3 * acc,s_Steer steer)
{
	int index = threadIdx.x  + (blockIdx.x * blockDim.x);
	
	if(index < N) 
	{
		glm::vec4 my_pos = pos[index];
		glm::vec3 my_vel = vel[index];
	
#if PART2 == 0
		glm::vec3 accel = ACC(N, my_pos, pos);
#else 
		glm::vec3 accel = flockAcc(N,my_pos,pos,vel,my_vel,steer);
#endif
		acc[index] = accel;
	}
}

__global__
void updateS(int N, float dt, glm::vec4 * pos, glm::vec3 * vel, glm::vec3 * acc)
{
	int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if(index < N)
	{
#if PART2 == 0
		vel[index] += acc[index]*dt;
#else
		vel[index] = acc[index] * vel_mag;
#endif
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
       // glm::vec3 color(0.05, 0.15, 0.3);
        //glm::vec3 acc = ACC(N, glm::vec4((x-w2)/c_scale_w,(y-h2)/c_scale_h,0,1), pos);
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
	
	/*stePointer = NULL;
	cudaMalloc((void**)&stePointer, sizeof(s_Steer));
	checkCUDAErrorWithLine("Kernel failed");*/

    generateRandomPosArray<<<fullBlocksPerGrid, blockSize>>>(1, numObjects, dev_pos, scene_scale, planetMass);
    checkCUDAErrorWithLine("Kernel failed!");
#if PART2 == 0
    generateCircularVelArray<<<fullBlocksPerGrid, blockSize>>>(2, numObjects, dev_vel, dev_pos);
#else
	generateRandomVelArray<<<fullBlocksPerGrid,blockSize>>>(2,numObjects,dev_vel,scene_scale,vel_mag);
#endif
    checkCUDAErrorWithLine("Kernel failed!");
	cudaThreadSynchronize();
}
__global__
void copyData(int N,glm::vec4* copypos,glm::vec4* pos)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if(index < N)
	{
		copypos[index] = pos[index];
	}
}
void cudaNBodyUpdateWrapper(float dt,int frame)
{
	dim3 fullBlocksPerGrid((int)ceil(float(numObjects)/float(blockSize)));
#if PART2 == 1
	thrust::device_ptr<glm::vec4> posEnd; 
	cudaMalloc((void**)&poscopy, numObjects*sizeof(glm::vec4));
    checkCUDAErrorWithLine("Kernel failed!");
	copyData<<<fullBlocksPerGrid,blockSize>>>(numObjects,poscopy,dev_pos);

    thrust::device_ptr<glm::vec4> posStart = thrust::device_pointer_cast(poscopy);
	int a = (steer.steeringForce <EPSILON);
	if(steer.steeringForce < EPSILON)
	{
		posEnd = thrust::remove_if(posStart,posStart + numObjects,isOutofRange());
		if((int)(posEnd-posStart) <= THREADHODE)
		{
			printf("all out of range");
			//need to do sth to make birds back
			// add steerForce
			steer.steeringForce = 4e7;
			//generate a random position in side a radius
			glm::vec3 randomTarget = generateRandomNumberFromThread(3, frame) - 0.5f;
			randomTarget = normalize(randomTarget);
			//
			//randomTarget = glm::vec3(randomTarget.x,randomTarget.y,0);
			float dist = generateRandomFloat(0,50,4,frame);
			steer.targetPos = dist * randomTarget;
			//steer.targetPos = glm::vec3(0,0,0);
		}
	}
	else
	{
		//find a time to cancel steeringforce
		posEnd = thrust::remove_if(posStart,posStart + numObjects,isIn());
		if((int)(posEnd-posStart) <= THREADHODE)
		{
			printf("In");
			//need to do sth to make birds back
			// add steerForce
			steer.steeringForce = 0;
		}
	}
	updateF<<<fullBlocksPerGrid, blockSize>>>(numObjects, dt, dev_pos, dev_vel,dev_acc,steer);
	checkCUDAErrorWithLine("Kernel failed!");
	updateS<<<fullBlocksPerGrid, blockSize>>>(numObjects, dt, dev_pos, dev_vel,dev_acc);
	{
    cudaError_t cudaerr = cudaDeviceSynchronize();
    if (cudaerr != CUDA_SUCCESS)
        printf("kernel launch failed with error \"%s\".\n",
               cudaGetErrorString(cudaerr));
	}
#else
	updateF<<<fullBlocksPerGrid, blockSize, blockSize*sizeof(glm::vec4)>>>(numObjects, dt, dev_pos, dev_vel, dev_acc,steer);
    checkCUDAErrorWithLine("Kernel failed!");
    updateS<<<fullBlocksPerGrid, blockSize>>>(numObjects, dt, dev_pos, dev_vel, dev_acc);
#endif

    checkCUDAErrorWithLine("Kernel failed!");
	cudaThreadSynchronize();
}

void cudaUpdateVBO(float * vbodptr, int width, int height)
{
    dim3 fullBlocksPerGrid((int)ceil(float(numObjects)/float(blockSize)));
    sendToVBO<<<fullBlocksPerGrid, blockSize>>>(numObjects, dev_pos, vbodptr, width, height, scene_scale);
	
   // checkCUDAErrorWithLine("Kernel failed!");
	cudaThreadSynchronize();
}

void cudaUpdatePBO(float4 * pbodptr, int width, int height)
{
    dim3 fullBlocksPerGrid((int)ceil(float(width*height)/float(blockSize)));
    sendToPBO<<<fullBlocksPerGrid, blockSize, blockSize*sizeof(glm::vec4)>>>(numObjects, dev_pos, pbodptr, width, height, scene_scale);
    cudaThreadSynchronize();
}
