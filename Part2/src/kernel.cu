#include <stdio.h>
#include <cuda.h>
#include <cmath>
#include "glm/glm.hpp"
#include "utilities.h"
#include "kernel.h"

#if RK == 1
    #define integration(x,y,z) rungekuttaInt(x,y,z)
#else
    #define integration(x,y,z) eulerInt(x,y,z)
#endif

//GLOBALS
dim3 threadsPerBlock(blockSize);

int numObjects;
//const float planetMass = 3e8;
//const __device__ float starMass = 5e10;
const __device__ float boidMass = 1;

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

__host__ __device__ 
float generateRandomNumberFromThread1D(float time, int index)
{
    thrust::default_random_engine rng(hash(index*time));
    thrust::uniform_real_distribution<float> u01(0,1);

    return (float) u01(rng);
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
        arr[index].z = rand.z; // 0.0f
        arr[index].w = mass;
    }
}

//Generate randomized starting velocities in the XY plane
__global__
void generateRandomVelArray(int time, int N, glm::vec3 * arr, float scale)
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if(index < N)
    {
        glm::vec3 rand = scale * (generateRandomNumberFromThread(time, index) - 0.5f);
        arr[index].x = rand.x;
        arr[index].y = rand.y;
        arr[index].z = rand.z;
    }
}

// REFERENCE for flocking: Craig's Paper http://www.red3d.com/cwr/papers/1999/gdc99steer.pdf
//           ALIGNMENT:  calculate the steering acceleration according to the velocity comparison 
//           COHESION:   calculate the steering acceleration according to the position comparison
//           SEPARATION: calculate the steering acceleration according to the 1/distance (or in other words is position comparison)

// TODO: Alignment for boids in a specific region of distance return the acceleration
__device__
glm::vec3 alignment(int N, glm::vec4 my_pos, glm::vec4 * their_pos, glm::vec3 my_vel, glm::vec3 * vel) {
    glm::vec3 steering_force;
    glm::vec3 average_vel(0.0f);
    unsigned int count = 0;

    // Align the boids in the region within the aligment distance
    for (unsigned int i = 0; i < N; ++ i ) {
      glm::vec3 dist_vec = glm::vec3(my_pos.x - their_pos[i].x, my_pos.y - their_pos[i].y, my_pos.z - their_pos[i].z);
      float distance = glm::length(dist_vec);
      if (distance > alignmentDistance || distance <= EPSILON || glm::dot (my_vel, glm::normalize(-dist_vec)) < cos(neighborAngle/2))
        continue;
      average_vel += vel[i];
      ++ count;
    }

    // Calculate the average velocity of all the boids in this region to find the steering acceleration
    if (count <= EPSILON)
      return glm::vec3(0.0f); 
    average_vel /= float(count);
    steering_force = average_vel - my_vel;        
    glm::vec3 steering_dir = glm::normalize(steering_force);

    // Clamp the force less than the maximum force
    steering_force = glm::length(steering_force) > maxForce ? (float) maxForce * steering_dir : (steering_force);
    return steering_force / boidMass;     
    //return glm::vec3(0.0f); 
}

// TODO: Cohesion for boids in a specific region of distance
__device__
glm::vec3 cohesion(int N, glm::vec4 my_pos, glm::vec4 * their_pos, glm::vec3 my_vel) {
    glm::vec3 steering_force;
    glm::vec3 average_pos(0.0f);
    unsigned int count = 0;

    // Make the boids cohere in the region within the cohesion distance
    for (unsigned int i = 0; i < N; ++ i ) {
      glm::vec3 dist_vec = glm::vec3(my_pos.x - their_pos[i].x, my_pos.y - their_pos[i].y, my_pos.z - their_pos[i].z);
      float distance = glm::length(dist_vec);
      if (distance > cohesionDistance || distance <= EPSILON || glm::dot (my_vel, glm::normalize(-dist_vec)) < cos(neighborAngle/2))
        continue;
      average_pos += glm::vec3(their_pos[i].x, their_pos[i].y, their_pos[i].z);
      ++ count;
    }

    // Calculate the average position of all the boids in this region to find the steering acceleration
    if (count <= EPSILON)
      return glm::vec3(0.0f); 
    average_pos /= float(count);
    steering_force = average_pos - glm::vec3(my_pos.x, my_pos.y, my_pos.w);
	
    // Clamp the force less than the maximum force
    glm::vec3 steering_dir = glm::normalize(steering_force);
    steering_force = glm::length(steering_force) > maxForce ? (float) maxForce * steering_dir : (steering_force);
    return steering_force / boidMass;
    // return glm::vec3(0.0f); 
}

// TODO: Separation for boids in a specific region of distance
__device__
glm::vec3 separation(int N, glm::vec4 my_pos, glm::vec4 * their_pos, glm::vec3 my_vel) {
    glm::vec3 steering_force(0.0f);

    // Separate the boids in the region within the separation distance
    for (unsigned int i = 0; i < N; ++ i ) {
      glm::vec3 dist_vec = glm::vec3(my_pos.x - their_pos[i].x, my_pos.y - their_pos[i].y, my_pos.z - their_pos[i].z);
      float distance = glm::length(dist_vec);
      if (distance > separationDistance || distance <= EPSILON || glm::dot (my_vel, glm::normalize(-dist_vec)) < cos(neighborAngle/2))
        continue;
      steering_force += dist_vec / pow(distance, 2);
    }
	
    // Clamp the force less than the maximum force
    glm::vec3 steering_dir = glm::normalize(steering_force);
    steering_force = glm::length(steering_force) > maxForce ? (float) maxForce * steering_dir : (steering_force);
    return steering_force / boidMass;
    // return glm::vec3(0.0f);     
}

// Simple Euler integration
__device__
void eulerInt(float dt, glm::vec4& pos, glm::vec3& vel) {
    pos.x += vel.x * dt;
    pos.y += vel.y * dt;
    pos.z += vel.z * dt;
}

__device__
void rungekuttaInt(float dt, glm::vec4& pos, glm::vec3& vel) {
    // Define the four increments for velocity according to Runge Kutta method, referring to http://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods
    glm::vec3 k1 = vel;                  // based on the slope at the beginning of the interval
    glm::vec3 k2 = k1 + 0.5f * dt * k1;  // based on the slope at the midpoint of the interval
    glm::vec3 k3 = k1 + 0.5f * dt * k2;  // based on the slope at the midpoint again
    glm::vec3 k4 = k1 + dt * k3;         // based on  the slope at the end of the interval
	
    // Final increment on the position
	glm::vec3 pos_inc = 1.0f / 6 * ( k1 + 2.0f * k2 + 2.0f * k3 + k4);

    // New position after RK4 integration
    pos.x += pos_inc.x * dt;
    pos.y += pos_inc.y * dt;
    pos.z += pos_inc.z * dt;
}

__device__
void borderHandle(glm::vec4& pos, glm::vec3& vel, float length, float width, float height) {
    if (pos.x > length / 2.0f) {
		pos.x = length / 2.0f;
		vel.x *= -1.0;
	} else if(pos.x < - length / 2.0f) {
      pos.x =  - length / 2.0f;
      vel.x *= -1.0f;
    }
    if (pos.y > width / 2.0f) {
      pos.y = width / 2.0f;
      vel.y *= -1.0f;
	} else if(pos.y <  - width / 2.0f) {
      pos.y =  - width / 2.0f;
	  vel.y *= -1.0f;
    }
    if (pos.z > height / 2.0f) {
      pos.z = height / 2.0f;
      vel.z *= -1.0f;
	} else if(pos.z < - height / 2.0f) {
      pos.z =  - height / 2.0f;
      vel.z *= -1.0f;
    }
}
//Update by integration scheme
__global__
void updateF(int N, float dt, glm::vec4 * pos, glm::vec3 * vel, glm::vec3 * acc)
{
    int index = threadIdx.x + (blockIdx.x * blockDim.x);
    glm::vec4 my_pos;
	glm::vec3 my_vel;
    glm::vec3 accel;

    if(index < N) {
      my_pos = pos[index];
	  my_vel = vel[index];
	}

	glm::vec3 weights(1.0f, 1.0f, 0.6f);
    glm::vec3 alignment_acc = alignment(N, my_pos, pos, my_vel, vel);
    glm::vec3 cohesion_acc = cohesion(N, my_pos, pos, my_vel);
    glm::vec3 separation_acc = separation(N, my_pos, pos, my_vel);
    accel = weights.x * alignment_acc + weights.y * cohesion_acc + weights.z * separation_acc;	 
    
	if(index < N) 
      acc[index] = accel;
} 

__global__
void updateS(int N, float dt, glm::vec4 * pos, glm::vec3 * vel, glm::vec3 * acc)
{
    int index = threadIdx.x + (blockIdx.x * blockDim.x);
    if( index < N ) {
      glm::vec3 dir = glm::normalize(vel[index] + acc[index] * dt);
      vel[index] = glm::length(vel[index]) * dir;
	  vel[index] = glm::length(vel[index]) > maxVelocity ? (float) maxVelocity * glm::normalize(vel[index]) : (vel[index]);
      integration(dt, pos[index], vel[index]);
    }
	borderHandle(pos[index], vel[index], 200.0f-10.0f, 200.0f-10.0f, 50.0f);
}


//Update the vertex buffer object
//(The VBO is where OpenGL looks for the positions for the planets)
__global__
void sendToVBO(int N, glm::vec4 * pos, glm::vec3 * vel, float * vbo, int width, int height, float s_scale)
{
    int index = threadIdx.x + (blockIdx.x * blockDim.x);

    float c_scale_w = -2.0f / s_scale;
    float c_scale_h = -2.0f / s_scale;

    if(index<N)
    {
        vbo[4*index+0] = pos[index].x*c_scale_w;
        vbo[4*index+1] = pos[index].y*c_scale_h;
        vbo[4*index+2] = vel[index].x*c_scale_w;
        vbo[4*index+3] = vel[index].y*c_scale_h;
    }
}

//Update the texture pixel buffer object
//(This texture is where openGL pulls the data for the height map)
__global__
void sendToPBO(int N, glm::vec4 * pos, glm::vec3 *vel, float4 * pbo, int width, int height, float s_scale)
{
    int index = threadIdx.x + (blockIdx.x * blockDim.x);
    int x = index % width;
    int y = index / width;
    float w2 = width / 2.0;
    float h2 = height / 2.0;

    float c_scale_w = width / s_scale;
    float c_scale_h = height / s_scale;

    glm::vec3 color(0.05, 0.15, 0.3);
	// Here I tried to use three kinds of field map the last and current one is with noise which refers to http://stackoverflow.com/questions/4200224/random-noise-functions-for-glsl
       //no height field: glm::vec3 acc = glm::vec3(0.0f, 0.0f, 0.0f);
       //refined grid height field with cohesion bump: glm::vec3 acc = (0.3f *1.0f/100.0f * cohesion(N, glm::vec4((x-w2)/c_scale_w,(y-h2)/c_scale_h,0,1), pos, glm::vec3(0)) * glm::vec3((sin(glm::fract(sin(glm::dot(glm::vec2((x-w2)/c_scale_w,(y-h2)/c_scale_h), glm::vec2(12.9898f,78.233f))) * 43758.5453f))-0.5f)*2.0f)); 
    
	// noise ocean wave height field
	glm::vec3 acc = 1.0f/50.0f*glm::vec3((sin(glm::fract(sin(glm::dot(glm::vec2((x-w2)/c_scale_w/4,(y-h2)/c_scale_h/4), glm::vec2(12.9898f,78.233f))) * 43758.5453f))-0.5f)*2.0f);
    if(x<width && y<height) {
      float mag = sqrt(sqrt(acc.x*acc.x + acc.y*acc.y + acc.z*acc.z));
      // Each thread writes one pixel location in the texture (textel)
	  float randPixelVal = generateRandomNumberFromThread1D(2, glm::length(pos[(int)((float)index * generateRandomNumberFromThread1D(index, 2))]));
      pbo[index].w = (mag < randPixelVal) ? mag * randPixelVal : 0.5f * randPixelVal;
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

    generateRandomPosArray<<<fullBlocksPerGrid, blockSize>>>(1, numObjects, dev_pos, scene_scale, boidMass);
    checkCUDAErrorWithLine("Kernel failed!");
	generateRandomVelArray<<<fullBlocksPerGrid, blockSize>>>(2, numObjects, dev_vel, scene_scale);
    checkCUDAErrorWithLine("Kernel failed!");
	cudaThreadSynchronize();
}

void cudaNBodyUpdateWrapper(float dt)
{
    dim3 fullBlocksPerGrid((int)ceil(float(numObjects)/float(blockSize)));
    updateF<<<fullBlocksPerGrid, blockSize, blockSize*sizeof(glm::vec4)>>>(numObjects, dt, dev_pos, dev_vel, dev_acc);
    checkCUDAErrorWithLine("Kernel failed!");
    updateS<<<fullBlocksPerGrid, blockSize>>>(numObjects, dt, dev_pos, dev_vel, dev_acc);
    checkCUDAErrorWithLine("Kernel failed!");
    cudaThreadSynchronize();
}

void cudaUpdateVBO(float * vbodptr, int width, int height)
{
    dim3 fullBlocksPerGrid((int)ceil(float(numObjects)/float(blockSize)));
    sendToVBO<<<fullBlocksPerGrid, blockSize>>>(numObjects, dev_pos, dev_vel, vbodptr, width, height, scene_scale);
    checkCUDAErrorWithLine("Kernel failed!");
    cudaThreadSynchronize(); 
}

 void cudaUpdatePBO(float4 * pbodptr, int width, int height)
 {
    dim3 fullBlocksPerGrid((int)ceil(float(width*height)/float(blockSize)));
    checkCUDAErrorWithLine("Kernel failed!");
    sendToPBO<<<fullBlocksPerGrid, blockSize, blockSize*sizeof(glm::vec4)>>>(numObjects, dev_pos, dev_vel, pbodptr, width, height, scene_scale);
    cudaThreadSynchronize();
 } 