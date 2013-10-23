#include <stdio.h>
#include <cuda.h>
#include <cmath>
#include "glm/glm.hpp"
#include "utilities.h"
#include "kernel.h"

#define CLOTHACC(x,y,z,w) clothAcc(x,y,z,w)

#ifdef SHARED
    #define ACC(x,y,z) sharedMemAcc(x,y,z)
	#define FLOCKVEL(x,y,z,w,a) sharedFlockVel(x,y,z,w,a)
#else
    #define ACC(x,y,z) naiveAcc(x,y,z)
	#define FLOCKVEL(x,y,z,w,a) naiveFlockVel(x,y,z,w,a)
#endif




//GLOBALS
dim3 threadsPerBlock(blockSize);

int numObjects;
glm::vec2 clothSize;
const float planetMass = 3e8;
const __device__ float starMass = 5e10;

int currentFrame;

const float scene_scale = 2e2; //size of the height map in simulation space

glm::vec4 * dev_pos;
glm::vec3 * dev_vel;

glm::vec3* springs;

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

__global__
	void generateKnotPos( int N, glm::vec4 * arr, float mass, glm::vec2 clothsize)
{
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	int index = x + (y * clothsize.x);

	if((x<clothsize.x) && (y<clothsize.y))
	{
		arr[index].x = ((float)x*2.0f-clothsize.x*1.0f)*96.0f/clothsize.x;
		arr[index].y = ((float)y*2.0f-clothsize.y*1.0f)*96.0f/clothsize.y;
		
		arr[index].z = -9.0f;//rand.z;
		if(x%10==0) arr[index].w=1;
		else if(y%10==0) arr[index].w=1;
		else arr[index].w=3;
	}
}

__global__
	void generateFlockPos(int N, glm::vec4 * arr)
{
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	if(index < N)
	{
		glm::vec3 rand = 200.0f*(generateRandomNumberFromThread(1, index)-0.5f);
		arr[index].x = rand.x;
		arr[index].y = rand.y;
		arr[index].z = rand.z;
		
		arr[index].w = index%5;
	}
}

__device__ bool isValidCoord(glm::vec2 clothsize, int x, int y)
{
	return (x>=0 && y>=0 && x<(int)clothsize.x && y<(int)clothsize.y);
}
__device__ int translateidx(glm::vec2 clothsize, int x, int y)
{
	return x+y*(int)clothsize.x;
}
__global__
	void generateSprings( int N, glm::vec4 * arr, glm::vec3* springs, glm::vec2 clothsize)
{
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	int index = x + (y * clothsize.x);

	if((x<clothsize.x) && (y<clothsize.y))
	{
		int idxoffset=index*12;		//
		int dx[12]={0,0,-1,1,-1-1,1,1,0,0,-2,2};
		int dy[12]={-1,1,0,0,-1,1,-1,1,-2,2,0,0};
		for(int i=0;i<STRUCTSPRINGS;i++)
		{
			int newx=x+dx[i];
			int newy=y+dy[i];
			int newidx=translateidx(clothsize,newx,newy);
		
			if(isValidCoord(clothsize,newx,newy))
			{
				springs[idxoffset+i]=glm::vec3(newidx,glm::distance(glm::vec3(arr[index]),glm::vec3(arr[newidx])),KStructure);
			}
			else
			{
				springs[idxoffset+i]=glm::vec3(0,0,-1);
			}
		}
		idxoffset+=STRUCTSPRINGS;
		for(int i=0;i<SHEARSPRINGS;i++)
		{
			int newx=x+dx[i+4];
			int newy=y+dy[i+4];
			int newidx=translateidx(clothsize,newx,newy);

			if(isValidCoord(clothsize,newx,newy))
			{
				springs[idxoffset+i]=glm::vec3(newidx,glm::distance(glm::vec3(arr[index]),glm::vec3(arr[newidx])),KShear);
			}
			else
			{
				springs[idxoffset+i]=glm::vec3(0,0,-1);
			}
		}
		idxoffset+=SHEARSPRINGS;
		for(int i=0;i<BENDSPRINGS;i++)
		{
			int newx=x+dx[i+8];
			int newy=y+dy[i+8];
			
			int newidx=translateidx(clothsize,newx,newy);

			if(isValidCoord(clothsize,newx,newy))
			{
				springs[idxoffset+i]=glm::vec3(newidx,glm::distance(glm::vec3(arr[index]),glm::vec3(arr[newidx])),KBend);
			}
			else
			{
				springs[idxoffset+i]=glm::vec3(0,0,-1);
			}
		}
	}
}
__global__
	void generateClothVelArray(int N, glm::vec3 * arr, glm::vec2 clothsize)
{
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	int index = x + (y * clothsize.x);
	if((x<clothsize.x) && (y<clothsize.y))
	{
		arr[index].x = 0;
		arr[index].y = 0;
		arr[index].z = 0;
	}
}

__global__
	void generateFlockVelArray(int N, glm::vec3 * arr)
{
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	if(index < N)
	{
		arr[index].x = 0;
		arr[index].y = 0;
		arr[index].z = 0;
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
	float r2=glm::distance(glm::vec3(us),glm::vec3(them));r2*=r2;
	if(r2<0.0001f) return glm::vec3(0,0,0);
	glm::vec3 direction=glm::normalize(glm::vec3(them-us));
	direction*=G*them.w/r2;
    return direction;
}

//TODO: Core force calc kernel global memory
__device__ 
glm::vec3 naiveAcc(int N, glm::vec4 my_pos, glm::vec4 * their_pos)
{
	glm::vec3 result(0,0,0);
	for(int i=0;i<N;i++)
	{
		 glm::vec3 acc = calculateAcceleration(my_pos, their_pos[i]);
		 result+=acc;
	}
	result+=calculateAcceleration(my_pos, glm::vec4(0,0,0,starMass));
    return result;
}
__device__ 
	glm::vec3 naiveFlockVel(int N, int myidx, glm::vec4 * their_pos, glm::vec3* velocities, glm::vec3 targetPos)
{
	glm::vec3 result(0,0,0);
	glm::vec3 cm(0,0,0);
	glm::vec3 avgVel(0,0,0);
	glm::vec3 separation(0,0,0);
	glm::vec3 myPos(their_pos[myidx]);

	int NofNeighbour=0;
	for(int i=0;i<N;i++)
	{
		glm::vec3 theirPos(their_pos[i]);
		float dist=glm::distance(myPos,theirPos);
		if(dist<RNeighbour)
		{
			NofNeighbour++;
			cm+=theirPos;		//for cohesion
			avgVel+=velocities[i];	//for alignment
			if(i!=myidx)
			{
				separation+=(myPos-theirPos)*(1.0f/dist/dist);
			}
		}
	}
	avgVel*=1.0f/float(NofNeighbour);
	cm*=1.0f/float(NofNeighbour);

	glm::vec3 randomNoise=generateRandomNumberFromThread(myPos.x*200.0f,myidx);
	randomNoise=randomNoise*2.0f-glm::vec3(1.0f,1.0f,1.0f);
	result+=randomNoise*KWander;
	result+=separation*KSeparation;
	result+=avgVel*KAlignment;
	result+=(cm-myPos)*KCohesion;

	result+=-(myPos-targetPos)*KArrival;

	if(glm::length(result)>VELTHRESHOLD) result=glm::normalize(result)*VELTHRESHOLD;
	return result;
}

__device__ 
	glm::vec3 sharedFlockVel(int N, int myidx, glm::vec4 * their_pos, glm::vec3* velocities, glm::vec3 targetPos)
{
	glm::vec3 result(0,0,0);
	glm::vec3 cm(0,0,0);
	glm::vec3 avgVel(0,0,0);
	glm::vec3 separation(0,0,0);
	glm::vec3 myPos(their_pos[myidx]);
	int NofNeighbour=0;
	int blocknum=(int)ceil(float(N)/float(blockSize));

	__shared__ glm::vec4 currentPos[blockSize];
	__shared__ glm::vec3 currentVel[blockSize];

	for(int i=0;i<blocknum;i++)
	{
		int targetIdx=threadIdx.x+i*blockDim.x;
		if(targetIdx<N)
		{
			currentPos[threadIdx.x]=their_pos[targetIdx];
			currentVel[threadIdx.x]=velocities[targetIdx];
		}
		__syncthreads();
		for(int j=0;j<blockDim.x;j++)
		{
			if(j+i*blockDim.x<N)
			{
				glm::vec3 theirPos(their_pos[j+i*blockDim.x]);
				float dist=glm::distance(myPos,theirPos);
				if(dist<RNeighbour)
				{
					NofNeighbour++;
					cm+=theirPos;		//for cohesion
					avgVel+=velocities[j+i*blockDim.x];	//for alignment
					if(j+i*blockDim.x!=myidx)
					{
						separation+=(myPos-theirPos)*(1.0f/dist/dist);
					}
				}
			}
		}
		__syncthreads();
	}

	
	avgVel*=1.0f/float(NofNeighbour);
	cm*=1.0f/float(NofNeighbour);

	glm::vec3 randomNoise=generateRandomNumberFromThread(myPos.x*200.0f,myidx);
	randomNoise=randomNoise*2.0f-glm::vec3(1.0f,1.0f,1.0f);
	result+=randomNoise*KWander;
	result+=separation*KSeparation;
	result+=avgVel*KAlignment;
	result+=(cm-myPos)*KCohesion;

	result+=-(myPos-targetPos)*KArrival;

	if(glm::length(result)>VELTHRESHOLD) result=glm::normalize(result)*VELTHRESHOLD;
	return result;
}

__device__ 
	glm::vec3 clothAcc(glm::vec2 clothsize, glm::vec2 myidx, glm::vec4 * their_pos, glm::vec3* springs)
{
	int x=myidx.x;
	int y=myidx.y;
	int index=translateidx(clothsize,x,y);

	glm::vec3 result(WINDX, WINDY ,SCALEDGRAVITY);
	glm::vec3 force(0,0,0);

	if(x>clothsize.x-4 )return glm::vec3(0,0,0);

	int idxoffset=index*SPRINGPERKNOT;		
	for(int i=0;i<SPRINGPERKNOT;i++)
	{
		int springidx=idxoffset+i;
		if(springs[springidx].z<0) continue;
		glm::vec3 pos1(their_pos[index]);
		glm::vec3 pos2(their_pos[(int)springs[springidx].x]);
		float nowlength=glm::distance(pos1,pos2);
		float standardlength=springs[springidx].y;
		float deltaL=-(standardlength-nowlength);
		if(abs(deltaL)<0.0001f) continue;
		force+=springs[springidx].z*deltaL*glm::normalize(glm::vec3(pos2-pos1));
	}
	result+=force;
	return result;
}


//TODO: Core force calc kernel shared memory
__device__ 
glm::vec3 sharedMemAcc(int N, glm::vec4 my_pos, glm::vec4 * their_pos)
{
	int index=threadIdx.x+blockIdx.x*blockDim.x;
	int blocknum=(int)ceil(float(N)/float(blockSize));
	glm::vec3 result(0,0,0);
	__shared__ glm::vec4 currentPos[blockSize];

	for(int i=0;i<blocknum;i++)
	{
		int targetIdx=threadIdx.x+i*blockDim.x;
		if(targetIdx<N)
			currentPos[threadIdx.x]=their_pos[targetIdx];
		__syncthreads();
		for(int j=0;j<blockDim.x;j++)
		{
			if(j+i*blockDim.x<N)	result+= calculateAcceleration(my_pos, currentPos[j]);
		}
		__syncthreads();
	}

    result+= calculateAcceleration(my_pos, glm::vec4(0,0,0,starMass));
    return result;
}


//Simple Euler integration scheme
__global__
void update(int N, float dt, glm::vec4 * pos, glm::vec3 * vel)
{
    int index = threadIdx.x + (blockIdx.x * blockDim.x);
    if( index < N )
    {
        glm::vec4 my_pos = pos[index];
        glm::vec3 acc = ACC(N, my_pos, pos);
		
        vel[index] += acc * dt;
        pos[index].x += vel[index].x * dt;
        pos[index].y += vel[index].y * dt;
        pos[index].z += vel[index].z * dt;

    }
}

__global__
	void flockupdate(int N, float dt, glm::vec4 * pos, glm::vec3 * vel, glm::vec3 targetPos)
{
	int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if( index < N )
	{
		glm::vec4 my_pos = pos[index];
		glm::vec3 newVel = FLOCKVEL(N, index, pos, vel,targetPos);
		glm::vec3 acc=newVel-vel[index];
		if(glm::length(acc)>5.0f) acc=glm::normalize(acc)*5.0f;

		vel[index]+=acc*dt;
		pos[index].x += vel[index].x * dt;
		pos[index].y += vel[index].y * dt;
		pos[index].z += vel[index].z * dt;
		if(abs(pos[index].x)>BOUNDARY|| abs(pos[index].y)>BOUNDARY || abs(pos[index].z)>BOUNDARY)
		{
			vel[index]=-vel[index];
			pos[index].x += 2*vel[index].x * dt;
			pos[index].y += 2*vel[index].y * dt;
			pos[index].z += 2*vel[index].z * dt;
		}

	}
}

__global__
void clothupdate(glm::vec2 clothsize, float dt, glm::vec4 * pos, glm::vec3 * vel, glm::vec3* springs)
{
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	int index = x + (y * clothsize.x);

	glm::vec3 acc(0,0,0);
	if((x<clothsize.x) && (y<clothsize.y))
	{
		for(int substeps=0;substeps<SUBSTEPS;substeps++)
		{
			glm::vec4 my_pos = pos[index];
			glm::vec3 acc = CLOTHACC(clothsize, glm::vec2(x,y), pos,springs);

			vel[index]*=0.9f;
			vel[index] += acc * dt;
			pos[index].x += vel[index].x * dt/(float(SUBSTEPS));
			pos[index].y += vel[index].y * dt/(float(SUBSTEPS));
			pos[index].z += vel[index].z * dt/(float(SUBSTEPS));
		}
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
        vbo[4*index+2]  = pos[index].z*c_scale_w;
        vbo[4*index+3] = pos[index].w;
    }
}

__global__
	void flockSendToVBO(int N, glm::vec4 * pos, float * vbo, int width, int height, float s_scale,glm::vec3 targetPos)
{
	int index = threadIdx.x + (blockIdx.x * blockDim.x);

	float c_scale_w = -2.0f / s_scale;
	float c_scale_h = -2.0f / s_scale;

	if(index<N)
	{
		vbo[4*index+0] = pos[index].x*c_scale_w;
		vbo[4*index+1] = pos[index].y*c_scale_h;
		vbo[4*index+2]  = pos[index].z*c_scale_h;
		vbo[4*index+3] = pos[index].w;
	}
	else if (index==N)
	{
		vbo[4*index+0] = -5000;
		vbo[4*index+1] = -5000;
		vbo[4*index+2]  = 0;
		vbo[4*index+3]  = 1;
	}
}

__global__
	void flockSendToVelBO(int N, glm::vec3 * vel, float * velbo, float s_scale)
{
	int index = threadIdx.x + (blockIdx.x * blockDim.x);

	float c_scale_w = -2.0f / s_scale;
	float c_scale_h = -2.0f / s_scale;

	if(index<N)
	{
		velbo[4*index+0] = vel[index].x*c_scale_w;
		velbo[4*index+1] = vel[index].y*c_scale_h;
		velbo[4*index+2]  = vel[index].z*c_scale_h;
		velbo[4*index+3] = 0;
	}
	else if (index==N)
	{
		velbo[4*index+0] = -5000;
		velbo[4*index+1] = -5000;
		velbo[4*index+2]  = 0;
		velbo[4*index+3]  = 1;
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

    if(x<width && y<height)
    {
        glm::vec3 color(0.05, 0.15, 0.3);
        //glm::vec3 acc = ACC(N, glm::vec4((x-w2)/c_scale_w,(y-h2)/c_scale_h,0,1), pos);
        //float mag = sqrt(sqrt(acc.x*acc.x + acc.y*acc.y + acc.z*acc.z));
        // Each thread writes one pixel location in the texture (textel)
        //pbo[index].w = (mag < 1.0f) ? mag : 1.0f;
		//pbo[index].x=x*c_scale_w;
		//pbo[index].y=y*c_scale_h;
		//pbo[index].z=0;
		//pbo[index].w = -2.9f;

    }
}

/*************************************
 * Wrappers for the __global__ calls *
 *************************************/

//Initialize memory, update some globals

void initCudaCloth(int width, int height)
{
	int N=width*height;
	numObjects=N;
	clothSize=glm::vec2(width,height);
	cudaMalloc((void**)&dev_pos, N*sizeof(glm::vec4));
	checkCUDAErrorWithLine("Kernel failed1!");
	cudaMalloc((void**)&dev_vel, N*sizeof(glm::vec3));
	checkCUDAErrorWithLine("Kernel failed2!");

	springs=NULL;
	cudaMalloc((void**)&springs,N*SPRINGPERKNOT*sizeof(glm::vec3));
	checkCUDAErrorWithLine("Kernel failed malloc springs!");
	
	
	dim3 threadsPerBlock(tilesize,tilesize);
	dim3 fullBlocksPerGrid((int)ceil(float(width)/float(tilesize)),(int)ceil(float(height)/float(tilesize)));

	generateKnotPos<<<fullBlocksPerGrid, threadsPerBlock>>>( numObjects, dev_pos, 1.0, glm::vec2(width,height));
	checkCUDAErrorWithLine("Kernel failed3!");
	generateSprings<<<fullBlocksPerGrid, threadsPerBlock>>>( numObjects, dev_pos, springs, glm::vec2(width,height));
	checkCUDAErrorWithLine("Kernel failed generating springs!");

	generateClothVelArray<<<fullBlocksPerGrid, threadsPerBlock>>>(numObjects, dev_vel, glm::vec2(width, height));
	checkCUDAErrorWithLine("Kernel failed generating initialVels!");

}

void initCudaFlock(int N)
{
	numObjects=N;
	
	cudaMalloc((void**)&dev_pos, N*sizeof(glm::vec4));
	checkCUDAErrorWithLine("Kernel failed1!");
	cudaMalloc((void**)&dev_vel, N*sizeof(glm::vec3));
	checkCUDAErrorWithLine("Kernel failed2!");

	dim3 fullBlocksPerGrid((int)ceil(float(N)/float(blockSize)));

	generateFlockPos<<<fullBlocksPerGrid, threadsPerBlock>>>( numObjects, dev_pos);
	checkCUDAErrorWithLine("Kernel failed3!");

	generateFlockVelArray<<<fullBlocksPerGrid, threadsPerBlock>>>(numObjects, dev_vel);
	checkCUDAErrorWithLine("Kernel failed generating initialVels!");

}

void initCuda(int N)
{
    numObjects = N;

    dim3 fullBlocksPerGrid((int)ceil(float(N)/float(blockSize)));

    cudaMalloc((void**)&dev_pos, N*sizeof(glm::vec4));
    checkCUDAErrorWithLine("Kernel failed!");
    cudaMalloc((void**)&dev_vel, N*sizeof(glm::vec3));
    checkCUDAErrorWithLine("Kernel failed!");

    generateRandomPosArray<<<fullBlocksPerGrid, blockSize>>>(1, numObjects, dev_pos, scene_scale, planetMass);
    checkCUDAErrorWithLine("Kernel failed!");
    generateCircularVelArray<<<fullBlocksPerGrid, blockSize>>>(2, numObjects, dev_vel, dev_pos);
    checkCUDAErrorWithLine("Kernel failed!");

}

void cudaNBodyUpdateWrapper(float dt)
{
    dim3 fullBlocksPerGrid((int)ceil(float(numObjects)/float(blockSize)));
    update<<<fullBlocksPerGrid, blockSize>>>(numObjects, dt, dev_pos, dev_vel);
    checkCUDAErrorWithLine("Kernel failed!");
}
void cudaFlockUpdateWrapper(float dt, glm::vec3 targetPos)
{

	dim3 fullBlocksPerGrid((int)ceil(float(numObjects)/float(blockSize)));
	flockupdate<<<fullBlocksPerGrid, blockSize>>>(numObjects, dt, dev_pos, dev_vel, targetPos);
	checkCUDAErrorWithLine("Kernel failed!");
}

void cudaClothUpdateWrapper(float dt)
{
	dim3 threadsPerBlock(tilesize,tilesize);
	dim3 fullBlocksPerGrid((int)ceil(float(clothSize.x)/float(tilesize)),(int)ceil(float(clothSize.y)/float(tilesize)));

	clothupdate<<<fullBlocksPerGrid, threadsPerBlock>>>(clothSize, dt, dev_pos, dev_vel,springs);
	checkCUDAErrorWithLine("Kernel failed!");
}


void cudaUpdateVBO(float * vbodptr, int width, int height,glm::vec3 targetPos)
{
    
    
#if SIMMODE==FLOCKSIM
	dim3 fullBlocksPerGrid((int)ceil(float(numObjects+1)/float(blockSize)));
	flockSendToVBO<<<fullBlocksPerGrid, blockSize>>>(numObjects, dev_pos, vbodptr, width, height, scene_scale,targetPos);
#else
	dim3 fullBlocksPerGrid((int)ceil(float(numObjects)/float(blockSize)));
	sendToVBO<<<fullBlocksPerGrid, blockSize>>>(numObjects, dev_pos, vbodptr, width, height, scene_scale);
#endif
    checkCUDAErrorWithLine("Kernel failed Update VBO!");
}

void cudaUpdateVelBO(float* velbodptr)
{

	dim3 fullBlocksPerGrid((int)ceil(float(numObjects)/float(blockSize)));
	flockSendToVelBO<<<fullBlocksPerGrid, blockSize>>>(numObjects, dev_vel, velbodptr, scene_scale);

}

void cudaUpdatePBO(float4 * pbodptr, int width, int height)
{
    dim3 fullBlocksPerGrid((int)ceil(float(width*height)/float(blockSize)));
    sendToPBO<<<fullBlocksPerGrid, blockSize>>>(numObjects, dev_pos, pbodptr, width, height, scene_scale);
    checkCUDAErrorWithLine("Kernel failed Update PBO!");
}
