#ifndef KERNEL_H
#define KERNEL_H

#include <stdio.h>
#include <thrust/random.h>
#include <cuda.h>
#include <cmath>

#if CUDA_VERSION >= 5000
    #include <helper_math.h>
#else
    #include <cutil_math.h>
#endif

#define blockSize 128
#define tilesize 16
#define checkCUDAErrorWithLine(msg) checkCUDAError(msg, __LINE__)
//#define SHARED


///BELOW ARE COEEFs AND PARAMs FOR CLOTH SIM
#define KStructure 200.55f
#define KShear 1.115f
#define KBend 8.515f
#define SCALEDGRAVITY 9.8f/2.0f
#define WINDX -3.0f
#define WINDY 0.0f

#define SUBSTEPS 12
#define STRUCTSPRINGS 4
#define SHEARSPRINGS 4
#define BENDSPRINGS 4
#define SPRINGPERKNOT 12


///BELOW ARE COEFFs AND PARAMs FOR FLOCK SIM
#define RNeighbour 30.0f
#define VELTHRESHOLD 15.0f
#define BOUNDARY 150.0f

#define KWander 1500.0f
#define KSeparation 950.0f
#define KCohesion 750.0f
#define KAlignment 550.0f
#define KArrival 00.0f

void checkCUDAError(const char *msg, int line);
void cudaNBodyUpdateWrapper(float dt);
void cudaFlockUpdateWrapper(float dt,glm::vec3 targetPos);
void cudaClothUpdateWrapper(float dt);
void initCudaCloth(int width, int height);
void initCuda(int N);
void initCudaFlock(int N);
void cudaUpdatePBO(float4 * pbodptr, int width, int height);
void cudaUpdateVBO(float * vbodptr, int width, int height,glm::vec3 targetPos);
void cudaUpdateVelBO(float* velbodptr);
#endif
