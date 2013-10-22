#include <stdio.h>
#include <cuda.h>
#include <cmath>
#include "glm/glm.hpp"
#include "utilities.h"
#include "kernel.h"

#if SHARED == 1
    #define ACC(x,y,z) sharedMemAcc(x,y,z)
#else
    #define ACC(x,y,z) naiveFlockVelocity(x,y,z,w)
#endif

//GLOBALS
dim3 threadsPerBlock(blockSize);

int numObjects;
const __device__  float planetMass = 3e8;
const __device__  float RNradius =50.0f;
const __device__  float KCohesion = 0.05f;
const __device__  float KAllign = 1.0f;
const __device__  float KArrival =1.0f;// 0.05f;
const __device__  float	KSeperate = 1.0f;
const __device__ float starMass = 5e10;

const float scene_scale = 2e2; //size of the height map in simulation space

glm::vec4 * dev_pos;
glm::vec3 * dev_vel;

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
	
	float r = sqrt((them.x-us.x)*(them.x-us.x) + (them.y-us.y)*(them.y -us.y )+ (them.z-us.z)*(them.z-us.z));
	glm::vec4 a = ((G * them.w * (them-us))/(pow(r,3)));//planetMass
    return glm::vec3(a.x,a.y,a.z);
}


//TODO: Core force calc kernel global memory
__device__ 
glm::vec3 naiveFlockVelocity(int N, glm::vec4 my_pos, glm::vec4 * their_pos, glm::vec3 my_vel,glm::vec3 * their_vel)
{
	glm::vec4 flock(0.0f,0.0f,0.0f,0.0f) ;
	int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if(index < N)
	{
	
	glm::vec3 m_pos(my_pos.x,my_pos.y,my_pos.z);
	
	float dist = 0.0f;
	int c=1;
	
	// Calculate separtion,cohesion and allignment velocity
	glm::vec4 vsp(0,0,0,0),coh(0,0,0,0),ali(0,0,0,0),ariv(0,0,0,0),tmp1(0,0,0,0),tmp2(0,0,0,0) ;
	ali.x = my_vel.x;
	ali.y = my_vel.y;
	ali.z = my_vel.z;

    coh.x = my_pos.x;
	coh.y = my_pos.y;
	coh.z = my_pos.z;

	for(int i=0; i<N ; i++)
	{
		glm::vec3 t_pos(their_pos[i].x,their_pos[i].y,their_pos[i].z);
	    if ( i != index)
		{
			dist = glm::length(t_pos - m_pos);
			if(dist < RNradius)
			{
				tmp1 = glm::normalize(glm::vec4((m_pos - t_pos),0.0f));
				vsp.x += tmp1.x;
				vsp.y += tmp1.y;
				vsp.z += tmp1.z;

				coh.x += t_pos.x;
				coh.y += t_pos.y;
				coh.z += t_pos.z;

				tmp2 = glm::vec4(their_vel[i],0.0f);//glm::normalize(glm::vec4(their_vel[i],0.0f));
				ali.x += tmp2.x;
				ali.y += tmp2.y;
				ali.z += tmp2.z;

				c++;
			}
		}
	}

	coh  = KCohesion * (coh/c- my_pos) ;
	ali  = KAllign   * glm::normalize((ali)/c);
	ariv = KArrival  * (glm::vec4(0,0,0,0) - glm::vec4(my_pos.x,my_pos.y,my_pos.z,0));
	vsp  = 0.0005f * ariv  + 1.0f * KSeperate * vsp ; 
	// Different ratios contribute to the final flocking velocity 
	flock = 0.2f * vsp  + 0.4f * coh + 0.6f * ali ;
	}
    return glm::vec3(flock.x,flock.y,flock.z);
}


//TODO: Core force calc kernel shared memory
__device__ 
glm::vec3 sharedMemAcc(int N, glm::vec4 my_pos, glm::vec4 * their_pos)
{
	__shared__ glm::vec4 sh_their_pos[25] ; //extern 
	int index = threadIdx.x + (blockIdx.x * blockDim.x);
	sh_their_pos[threadIdx.x] = their_pos[index];
    glm::vec3 acc(0.0f,0.0f,0.0f) ;
	for(int i=0; i<N ; i++)
	{
		if(my_pos != their_pos[i])
			acc += calculateAcceleration(my_pos, sh_their_pos[i]);
	}
	acc += calculateAcceleration(my_pos, glm::vec4(0,0,0,starMass));
    return acc;
}


//Simple Euler integration scheme
__global__
void update(int N, float dt, glm::vec4 * pos, glm::vec3 * vel)
{
    int index = threadIdx.x + (blockIdx.x * blockDim.x);
    if( index < N )
    {
        glm::vec4 my_pos = pos[index];
        vel[index] = naiveFlockVelocity(N, my_pos, pos,vel[index],vel);// naiveAcc(N, my_pos, pos);//  ACC(N, my_pos, pos);//  glm::vec3(0,0,1);//
       
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

    if(x<width && y<height)
    {
        glm::vec3 color(0.05, 0.15, 0.3);
        glm::vec3 acc =glm::vec3(1,1,1);// ACC(N, glm::vec4((x-w2)/c_scale_w,(y-h2)/c_scale_h,0,1), pos);
        float mag = 0.02f;//sqrt(sqrt(acc.x*acc.x + acc.y*acc.y + acc.z*acc.z));
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

    generateRandomPosArray<<<fullBlocksPerGrid, blockSize,N>>>(1, numObjects, dev_pos, scene_scale, planetMass);
    checkCUDAErrorWithLine("Kernel failed!");
    generateCircularVelArray<<<fullBlocksPerGrid, blockSize,N>>>(2, numObjects, dev_vel, dev_pos);
    checkCUDAErrorWithLine("Kernel failed!");
}

void cudaNBodyUpdateWrapper(float dt, int N)
{
    dim3 fullBlocksPerGrid((int)ceil(float(numObjects)/float(blockSize)));
    update<<<fullBlocksPerGrid, blockSize,25>>>(numObjects, dt, dev_pos, dev_vel);
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
