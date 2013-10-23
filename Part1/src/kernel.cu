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

#define FLOCKING 0

//GLOBALS
dim3 threadsPerBlock(blockSize);

int numObjects;
const float planetMass = 3e8;
const __device__ float starMass = 5e9;

const float scene_scale = 2e1; //size of the height map in simulation space

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
        //arr[index].z = 0.0f;//rand.z;
        arr[index].z = rand.z;
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
        glm::vec3 D = 10.0f*glm::normalize(glm::cross(R/r,glm::vec3(0,0,1)));
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
	arr[index].z = 0.0;
        //arr[index].z = rand.z;
    }
}

//DONE: Determine force between two bodies
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

    float m_them = them.w; 
   
    glm::vec3 d = glm::vec3(us.x, us.y, us.z) - glm::vec3(them.x, them.y, them.z); 
    float r2 = glm::dot( d, d );

    // EPSILON softening-factor
    float a = -G*m_them/(r2 + 1e-1);
    return a*glm::normalize( d );
}

//DONE: Core force calc kernel global memory
__device__ 
glm::vec3 naiveAcc(int N, glm::vec4 my_pos, glm::vec4 * their_pos)
{
    int index = threadIdx.x + (blockIdx.x * blockDim.x);
    // Calculate acceleration from star
    glm::vec3 acc = calculateAcceleration(my_pos, glm::vec4(0,0,0,starMass));
    // Calculate accelerations from other planets
    for ( int i=0; i < N; ++i ) {
      if ( i == index ) 
	continue;
      acc += calculateAcceleration(my_pos, their_pos[i]);
    }
    return acc;
}


//DONE: Core force calc kernel shared memory
__device__ 
glm::vec3 sharedMemAcc(int N, glm::vec4 my_pos, glm::vec4 * their_pos)
{
    __shared__ glm::vec4 shared_their_pos[blockSize];
    glm::vec3 acc = calculateAcceleration(my_pos, glm::vec4(0,0,0,starMass));

    int index = threadIdx.x + (blockIdx.x * blockDim.x);
    // Copy a segment of positions from global to shared memory
    int num_iter = 0;
    for ( int i = 0; i < N; i += blockDim.x ) { 
      // Compute global memory index to pull in
      int gbl_index = threadIdx.x + (num_iter * blockDim.x);
      num_iter++;
       
      shared_their_pos[threadIdx.x] = their_pos[gbl_index];
      // Don't forget to sync after the copy 
      __syncthreads();
      

      // Calculate accelerations from other planets using from shared mem
      for ( int j=0; j < blockDim.x; j++ ) {
	  if ( i+j != index ) 
	    acc += calculateAcceleration(my_pos, shared_their_pos[j]);
      }
      // Sync before next copy
      __syncthreads();
    }
    return acc;
    //return calculateAcceleration(my_pos, glm::vec4(0,0,0,starMass));
}

//DONE
__device__
glm::vec3 Alignment( int N, glm::vec4 my_pos, glm::vec4* pos, glm::vec3* vel )
{
  glm::vec3 ave_vel;
  float r2;
  glm::vec3 d;
  int index = threadIdx.x + (blockIdx.x * blockDim.x);
  // Compute average velocity
  // Compute average position
  int cnt;
  for ( int i=0; i < N; ++i ) {
    if ( i == index ) 
      continue;
    d = glm::vec3( pos[i].x-my_pos.x, pos[i].y-my_pos.y, pos[i].z-my_pos.z); 
    r2 = glm::dot( d, d );
    if ( r2 < 5.0 ) {
      ave_vel += glm::vec3( vel[i].x, vel[i].y, vel[i].z );
      cnt ++;
    }
  }
  ave_vel= ave_vel/float(cnt);

  return ave_vel;
  //return glm::vec3(0.0, 0.0, 0.0);
} 

//DONE 
__device__
glm::vec3 Cohesion( int N, glm::vec4 my_pos, glm::vec4* pos)
{
  glm::vec3 ave_pos;
  float r2;
  glm::vec3 d;
  int index = threadIdx.x + (blockIdx.x * blockDim.x);
  int cnt = 0;
  // Compute average position weighted by distance
  for ( int i=0; i < N; ++i ) {
    if ( i == index ) 
      continue;
    d = glm::vec3( pos[i].x-my_pos.x, pos[i].y-my_pos.y, pos[i].z-my_pos.z); 
    r2 = glm::dot( d, d );
    if ( r2 < 5.0 ) {
      cnt++;
      ave_pos += glm::vec3( pos[i].x, pos[i].y, pos[i].z );
    }
  }
  ave_pos = ave_pos/float(cnt);

  d = glm::vec3(ave_pos.x-my_pos.x, ave_pos.y-my_pos.y, ave_pos.z-my_pos.z); 

  //float r = glm::length(d) + EPSILON;
  //float s = sqrt(1.0f/r);
  //glm::vec3 D = glm::normalize(glm::cross(d/r,glm::vec3(0,0,1)));

  //return s*D;

  return glm::normalize(d);
} 

//DONE
__device__
glm::vec3 Seperation( int N, glm::vec4 my_pos, glm::vec4* pos  )
{

  int index = threadIdx.x + (blockIdx.x * blockDim.x);

  glm::vec3 acc;
  glm::vec3 d;
  float r2;
  // Compute repulsion force
  for ( int i=0; i < N; ++i ) {
    if ( i == index ) 
      continue;
    d = glm::vec3( pos[i].x-my_pos.x, pos[i].y-my_pos.y, pos[i].z-my_pos.z); 
    r2 = glm::dot( d, d );
    if ( r2 < 1.0 ) 
      acc += -glm::normalize(d);
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

//Simple Euler integration scheme
__global__
void updateS(int N, float dt, glm::vec4 * pos, glm::vec3 * vel, glm::vec3 * acc, glm::vec3 star_position )
{
    int index = threadIdx.x + (blockIdx.x * blockDim.x);
    if( index < N )
    {
        glm::vec4 my_pos = pos[index];

	#if FLOCKING == 0 
        //glm::vec3 acc = ACC(N, my_pos, pos);
        vel[index] += acc[index] * dt;
	#else 
	// Align velocity with flock average
	glm::vec3 align_vel = Alignment( N, my_pos, pos, vel );
	// Attract towards flock average
	glm::vec3 cohesion_vel = Cohesion( N, my_pos, pos );
	// Repel from nearby objects
	glm::vec3 seperation_vel = Seperation( N, my_pos, pos );

	glm::vec3 weights = glm::vec3( 0.8, 0.2, 0.5 );	
	// Need some weights
	vel[index] = weights.x*align_vel + weights.y*cohesion_vel + weights.z*seperation_vel;

	// Add in circular velocity around star
	//glm::vec3 star_position( 1.0, 0.0, 0.0 );
        glm::vec3 R = glm::vec3(pos[index].x-star_position.x, pos[index].y-star_position.y, pos[index].z-star_position.z);
        float r = glm::length(R) + EPSILON;
        float s = sqrt(G*starMass/r);
        glm::vec3 D = glm::normalize(glm::cross(R/r,glm::vec3(0,0,1)));
	vel[index] += 1.0f*D;

	// Add in attractive velocity toward star
	vel[index] += -0.01f*R;

	// Add in damping
	//vel[index] *= 0.7f;

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

    if(x<width && y<height)
    {
        glm::vec3 color(0.05, 0.15, 0.3);
	glm::vec3 acc = glm::vec3( 0.0, 0.0, 0.5 );
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

    printf("Num Objects: %d \n", numObjects );
    printf("Num blocks: %d, BlockSize, %d \n", fullBlocksPerGrid.x, blockSize );
    printf("Shared: %d \n", SHARED );

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
}

void cudaNBodyUpdateWrapper(float dt, glm::vec3 goal_position )
{
    dim3 fullBlocksPerGrid((int)ceil(float(numObjects)/float(blockSize)));
    updateF<<<fullBlocksPerGrid, blockSize>>>(numObjects, dt, dev_pos, dev_vel, dev_acc);
    updateS<<<fullBlocksPerGrid, blockSize>>>(numObjects, dt, dev_pos, dev_vel, dev_acc, goal_position);
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
