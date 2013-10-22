#ifndef KERNEL_H
#define KERNEL_H

#include <stdio.h>
#include <thrust/random.h>
#include <cuda.h>
#include <cmath>
#include "glFields.h"

#if CUDA_VERSION >= 5000
    #include <helper_math.h>
#else
    #include <cutil_math.h>
#endif

#define blockSize 128
#define checkCUDAErrorWithLine(msg) checkCUDAError(msg, __LINE__)

//TODO: Add parameterized control for each boid having different rules.
struct BoidProps{
	glm::vec3 pos;
	glm::vec3 heading;//Direction and magnitude separated 
	float speed;
	float rollAngle;
};

struct WorldProps{
	glm::vec3 InitialDims;//Represents entire length
	glm::vec3 WorldBounds;//Max distance from origin boids can travel in +/-x, +/-y, +z; Boids can't fly negative Z
	float InitialMaxVel;
	float GroundAvoidanceHeight;
	float GroundAvoidanceForce;//N
	float WallStiffness;//N/m
	float MinSpeed;
	float BarrelRoll;//Forced roll.
	float ViewAngleCos;
	glm::vec3 AttractionZone;//(minDist, maxDist, force);
	glm::vec3 AlignmentZone;//(minDist, maxDist, force);
	glm::vec3 RepulsionZone;//(minDist, maxDist, force);
};


void checkCUDAError(const char *msg, int line);
void cudaNBodyUpdateWrapper(float dt, WorldProps world);
void initCuda(int N, WorldProps world);
void cudaUpdateVBO(float * vbodptr);
#endif
