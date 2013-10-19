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
const __device__ float starMass = 5e10;
const __device__ float boidMass = 5;

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
        arr[index].z = rand.z; // 0.0f
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
        arr[index].z = rand.z;
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

    // Load the data
    glm::vec3 us_pos(us.x, us.y, us.z);
    glm::vec3 them_pos(them.x, them.y, them.z);
    float m_them = them.w;
	
    // Calculate the distance and direction between the two objects (Note: the direction is them - us).
    glm::vec3 r = them_pos - us_pos;
    float distance = glm::length(r);
    glm::vec3 direction = glm::normalize(r);
	
    // Return no acceleration when the two bodies are very close to each other
    if (abs(distance) < EPSILON)
	  return glm::vec3(0.0f);

    // Calculate the acceleration between two bodies using Newton's Law of Universal Gravitation, referring to http://en.wikipedia.org/wiki/Newton%27s_law_of_universal_gravitation
    glm::vec3 acc = direction * (float)G * m_them / pow(distance,2);
    return acc;
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
      float distance = glm::distance(my_pos, their_pos[i]);
      if (distance > alignmentDistance || distance <= EPSILON)
        continue;
      average_vel += glm::vec3(vel[i].x, vel[i].y, vel[i].z);
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
glm::vec3 cohesion(int N, glm::vec4 my_pos, glm::vec4 * their_pos) {
    glm::vec3 steering_force;
    glm::vec3 average_pos(0.0f);
    unsigned int count = 0;

    // Make the boids cohere in the region within the cohesion distance
    for (unsigned int i = 0; i < N; ++ i ) {
      float distance = glm::distance(my_pos, their_pos[i]);
      if (distance > cohesionDistance || distance <= EPSILON)
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
glm::vec3 separation(int N, glm::vec4 my_pos, glm::vec4 * their_pos) {
    glm::vec3 steering_force(0.0f);

    // Separate the boids in the region within the separation distance
    for (unsigned int i = 0; i < N; ++ i ) {
      float distance = glm::distance(my_pos, their_pos[i]);
      if (distance > separationDistance || distance <= EPSILON)
        continue;
      steering_force += glm::vec3(my_pos.x - their_pos[i].x, my_pos.y - their_pos[i].y, my_pos.z - their_pos[i].z) / pow(distance, 2);
	  //steering_force += -glm::normalize(glm::vec3(my_pos.x - their_pos[i].x, my_pos.y - their_pos[i].y, my_pos.z - their_pos[i].z) );
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

//Update by integration scheme
__global__
void update(int N, float dt, glm::vec4 * pos, glm::vec3 * vel)
{
    int index = threadIdx.x + (blockIdx.x * blockDim.x);
    if( index < N )
    {
      glm::vec4 my_pos = pos[index];
	  glm::vec3 my_vel = vel[index];
	  glm::vec3 weights = glm::vec3(1.0f, 1.0f, 0.6f);
	  glm::vec3 alignment_acc = alignment(N, my_pos, pos, my_vel, vel);
	  glm::vec3 cohesion_acc = cohesion(N, my_pos, pos);
	  glm::vec3 separation_acc = separation(N, my_pos, pos);
      glm::vec3 acc = weights.x * alignment_acc + weights.y * cohesion_acc + weights.z * separation_acc;	  
	  vel[index] += acc * dt;
      integration(dt, pos[index], vel[index]);
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
		glm::vec3 acc = glm::vec3( 1.0, 1.0, 0.5 );
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

    generateRandomPosArray<<<fullBlocksPerGrid, blockSize>>>(1, numObjects, dev_pos, scene_scale, boidMass);
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
