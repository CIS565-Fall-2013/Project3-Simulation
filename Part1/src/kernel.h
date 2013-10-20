#ifndef KERNEL_H
#define KERNEL_H

#include <stdio.h>
#include <thrust/random.h>
#include <thrust/device_ptr.h>
#include <thrust/remove.h>
#include <cuda.h>
#include <cmath>

#if CUDA_VERSION >= 5000
    #include <helper_math.h>
#else
    #include <cutil_math.h>
#endif

#define blockSize 128
#define checkCUDAErrorWithLine(msg) checkCUDAError(msg, __LINE__)
#define SHARED 0
#define PART2 1
#define NEIGHBOR			3000
#define SNEIGHBOR			100
#define VIEWANGLE			180.0/180.0*PI
#define FLOCKFORCE			6e7
#define COHESIONFORCE		2e7
#define SEPARATIONFORCE		2e7
#define RANGE				300.0
#define MINRANGE			300.0
#define THREADHODE			0 //THIS IS FOR THE CHECK OF WHETHER BIRDS ARE WITHIN THE RANGE
void checkCUDAError(const char *msg, int line);
void cudaNBodyUpdateWrapper(float dt,int frame);
void initCuda(int N);
void cudaUpdatePBO(float4 * pbodptr, int width, int height);
void cudaUpdateVBO(float * vbodptr, int width, int height);

struct s_Steer{
	__host__ __device__ s_Steer()
	{
		targetPos = glm::vec3(0,0,0);
		steeringForce = 0;
	}
	glm::vec3 targetPos;
	float steeringForce;
};

struct isOutofRange{
	__host__ __device__
	bool operator()(const glm::vec4 v)
	{
		return (glm::length(glm::vec3(v.x,v.y,v.z)) > RANGE);
	}
};
struct isIn{
	__host__ __device__
		bool operator()(const glm::vec4 v)
	{
		return (glm::length(glm::vec3(v.x,v.y,v.z)) <= MINRANGE);
	}
};

#endif
