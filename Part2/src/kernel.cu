#include <stdio.h>
#include <cuda.h>
#include <cmath>
#include "glm/glm.hpp"
#include "utilities.h"
#include "kernel.h"

//GLOBALS
const glm::vec3 gravity(0, -9.8, 0);
int dimx, dimz;
dim3 threadsPerBlock;
dim3 fullBlocksPerGridV; //for parallel vertex operations
dim3 fullBlocksPerGridT; //for parallel triangle operations
int threadsPerBlockE; //for parallel edge operations
int fullBlocksPerGridE; //for parallel edge operations

int vertexCount;
int triangleCount;
int edgeCount;

glm::vec3* pos;
glm::vec3* predicted_pos;
glm::vec3* normals;
glm::vec3* vel;
bool* lock_pos;
float* inv_mass;
//TEMP
//glm::vec3* acc;
//float Ks = 0.01;
//float Kd = 0;

int* triangles;
Edge* edges;

int solver_iterations = 10;
float fp_stiff = 1;
<<<<<<< HEAD
float st_stiff = 0.9;
=======
float st_stiff = 0.5;
>>>>>>> 064d51024169ffa5eeb020a4b68f4ae59822a0e2
float b_stiff = 0.05;
float col_stiff = 1;

FixedPointConstraint* fp_constraints;
StretchConstraint* st_constraints;
BendConstraint* b_constraints;
CollisionConstraint* col_onstraints;
//SelfCollisionConstraint* sc_constraints;

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
        //exit(EXIT_FAILURE); 
    }
} 

// Initialize vertex attributes
__global__ void initVertexAttributes(glm::vec3* pos, glm::vec3* predicted_pos, glm::vec3* vel,
									 glm::vec3* normals, bool* lock_pos, float* inv_mass,
									 int dimx, int dimz, float dx, float dz, float y0, float mass)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	
	if (i < dimx && j < dimz) {
		int index = j + i * dimz;

		pos[index] = glm::vec3(i * dx, y0, j * dz);
		predicted_pos[index] = glm::vec3(i * dx, y0, j * dz);
		vel[index] = glm::vec3(0, 0, 0);
		normals[index] = glm::vec3(0, 1, 0);
		lock_pos[index] = false;
		inv_mass[index] = 1/mass;
	}
}

// Populate the triangle list
__global__ void initTriangles(int* triangles, Edge* edges, int dimx, int dimz)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	
	if (i < dimx-1 && j < dimz-1) {
		int index = j + i * (dimz-1);

		triangles[6*index  ] = dimz*i + j;
		triangles[6*index+1] = dimz*i + j + 1;
		triangles[6*index+2] = dimz*(i+1) + j;
		triangles[6*index+3] = dimz*(i+1) + j;
		triangles[6*index+4] = dimz*i + j + 1;
		triangles[6*index+5] = dimz*(i+1) + j + 1;

		edges[3*index  ] = Edge(dimz*i+j, dimz*i+j+1, 6*index, 6*((i-1)*dimz+j)+3);
		edges[3*index+1] = Edge(dimz*i+j+1, dimz*(i+1)+j, 6*index, 6*index+3);
		edges[3*index+2] = Edge(dimz*(i+1)+j, dimz*i+j, 6*index, 6*(i*dimz+j-1)+3);
		if (i == 0) {
			edges[3*index].tri2 = -1; //no triangle on the other side
		}
		if (j == 0) {
			edges[3*index+2].tri2 = -1; //no triangle on the other side
		}

		if (i == dimx-2) {
			edges[3*(dimx-1)*(dimz-1)+j] = Edge(dimz*(i+1)+j+1, dimz*(i+1)+j, 6*index+3, -1);
		}
		if (j == dimz-2) {
			edges[3*(dimx-1)*(dimz-1)+dimz-1+i] = Edge(dimz*i+j+1, dimz*(i+1)+j+1, 6*index+3, -1);
		}
	}
}

// Apply external force, damp velocity and compute predicted positions
__global__ void preConstraintsUpdate(glm::vec3* predicted_pos, glm::vec3* vel, float* inv_mass,
									 int dimx, int dimz, glm::vec3 force, float dt)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	
	if (i < dimx && j < dimz) {
		int index = j + i * dimz;
		vel[index] += force * inv_mass[index] * dt;
		// no damp velocity for now
		predicted_pos[index] += vel[index] * dt;
	}
}

// Generate fixed point constraints
__global__ void generateFPConstraints(FixedPointConstraint* fp_constraints,
									  int dimx, int dimz, float dx, float dz, float y0)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;

	if (i < dimx && j < dimz) {
		int index = j + i * dimz;

		//if (i == 0 && j == 0) {
		//	glm::vec3 fp(0, y0, 0);
		//	fp_constraints[index] = FixedPointConstraint(index, fp);
		//}
		//else if (i == 0 && j == dimz-1) {
		//	glm::vec3 fp(0, y0, (dimz-1)*dz);
		//	fp_constraints[index] = FixedPointConstraint(index, fp);
		//}
		if (i == 0) {
			glm::vec3 fp(0, y0, dz*j);
			fp_constraints[index] = FixedPointConstraint(index, fp);
		}
		else {
			fp_constraints[index] = FixedPointConstraint(-1, glm::vec3(0));
		}
	}
}

__global__ void generateStretchBendConstraints(StretchConstraint* st_constraints, BendConstraint* b_constraints,
											   glm::vec3* pos, Edge* edges, int* triangles, int edgeCount)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index < edgeCount) {
		// generate stretch constraints
		int v1 = edges[index].v1;
		int v2 = edges[index].v2;
		glm::vec3 p1 = pos[v1];
		glm::vec3 p2 = pos[v2];
		float l = glm::length(p1 - p2);
		st_constraints[index] = StretchConstraint(v1, v2, l);
	}
}

// Project fixed point constraints
__global__ void resolveFPConstraints(FixedPointConstraint* fp_constraints, glm::vec3* predicted_pos,
									 bool* lock_pos, float stiff, int dimx, int dimz)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;

	if (i < dimx && j < dimz) {
		int index = j + i * dimz;

		FixedPointConstraint fpc = fp_constraints[index];
		if (fpc.v0 != -1) {
			lock_pos[fpc.v0] = true;
			glm::vec3 dist = fpc.fixed_point - predicted_pos[fpc.v0];
			predicted_pos[fpc.v0] += dist * stiff;
		}
	}
}

__global__ void resolveStretchBendConstraints(StretchConstraint* st_constraints, BendConstraint* b_constraints,
											  glm::vec3* predicted_pos, float* inv_mass, bool* lock_pos,
											  float st_stiff, float b_stiff, int edgeCount)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index < edgeCount) {
		// project stretch constraints
		StretchConstraint stc = st_constraints[index];
		glm::vec3 sp1 = predicted_pos[stc.v1];
		glm::vec3 sp2 = predicted_pos[stc.v2];
		float l = glm::length(sp1 - sp2);

		float sw1 = lock_pos[stc.v1] ? 0 : inv_mass[stc.v1];
		float sw2 = lock_pos[stc.v2] ? 0 : inv_mass[stc.v2];
		if (!(sw1 < FLT_EPSILON && sw2 < FLT_EPSILON)) {
			glm::vec3 dist1 = -sw1 / (sw1+sw2) * (l-stc.rest_length) * glm::normalize(sp1-sp2);
			glm::vec3 dist2 = sw2 / (sw1+sw2) * (l-stc.rest_length) * glm::normalize(sp1-sp2);
			atomicAdd(&(predicted_pos[stc.v1].x), dist1.x * st_stiff);
			atomicAdd(&(predicted_pos[stc.v1].y), dist1.y * st_stiff);
			atomicAdd(&(predicted_pos[stc.v1].z), dist1.z * st_stiff);
			atomicAdd(&(predicted_pos[stc.v2].x), dist2.x * st_stiff);
			atomicAdd(&(predicted_pos[stc.v2].y), dist2.y * st_stiff);
			atomicAdd(&(predicted_pos[stc.v2].z), dist2.z * st_stiff);
		}
	}
}

//__global__ void resetAcc(glm::vec3* acc, int dimx, int dimz)
//{
//	int i = blockIdx.x * blockDim.x + threadIdx.x;
//	int j = blockIdx.y * blockDim.y + threadIdx.y;
//
//	if (i < dimx && j < dimz) {
//		int index = j + i * dimz;
//
//		acc[index] = glm::vec3(0, 0, 0);
//	}
//}
//
//__global__ void computeAcc(glm::vec3* acc, StretchConstraint* st_constraints, glm::vec3* predicted_pos,
//						   glm::vec3* vel, float* inv_mass, float Ks, float Kd, int edgeCount)
//{
//	int index = blockIdx.x * blockDim.x + threadIdx.x;
//
//	if (index < edgeCount) {
//		// project stretch constraints
//		StretchConstraint stc = st_constraints[index];
//		glm::vec3 sp1 = predicted_pos[stc.v1];
//		glm::vec3 sp2 = predicted_pos[stc.v2];
//		glm::vec3 sv1 = vel[stc.v1];
//		glm::vec3 sv2 = vel[stc.v2];
//		float l = glm::length(sp1 - sp2);
//
//		glm::vec3 acc1 = -Ks * (l-stc.rest_length) * glm::normalize(sp1-sp2) - Kd * (sv1-sv2)
//			* glm::normalize(sp1-sp2) * glm::normalize(sp1-sp2);
//		glm::vec3 acc2 = -Ks * (l-stc.rest_length) * glm::normalize(sp2-sp1) - Kd * (sv2-sv1)
//			* glm::normalize(sp2-sp1) * glm::normalize(sp2-sp1);
//		acc1 = acc1 * inv_mass[stc.v1];
//		acc2 = acc2 * inv_mass[stc.v2];
//
//		atomicAdd(&(acc[stc.v1].x), acc1.x);
//		atomicAdd(&(acc[stc.v1].y), acc1.y);
//		atomicAdd(&(acc[stc.v1].z), acc1.z);
//		atomicAdd(&(acc[stc.v2].x), acc2.x);
//		atomicAdd(&(acc[stc.v2].y), acc2.y);
//		atomicAdd(&(acc[stc.v2].z), acc2.z);
//	}
//}
//
//__global__ void applyAcc(glm::vec3* acc, glm::vec3* vel, glm::vec3* predicted_pos, bool* lock_pos,
//						 float dt, int dimx, int dimz)
//{
//	int i = blockIdx.x * blockDim.x + threadIdx.x;
//	int j = blockIdx.y * blockDim.y + threadIdx.y;
//
//	if (i < dimx && j < dimz) {
//		int index = j + i * dimz;
//
//		if (!lock_pos[index]) {
//			predicted_pos[index] += (vel[index] + acc[index]*dt) * dt;
//		}
//	}
//}


// Update velocity and position based on predicted position
__global__ void integrate(glm::vec3* pos, glm::vec3* predicted_pos, glm::vec3* vel, float dt, int dimx, int dimz)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;

	if (i < dimx && j < dimz) {
		int index = j + i * dimz;

		vel[index] = (predicted_pos[index] - pos[index]) / dt;
		pos[index] = predicted_pos[index];
	}
}

__global__ void resetNormal(glm::vec3* normals, int dimx, int dimz)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;

	if (i < dimx && j < dimz) {
		int index = j + i * dimz;

		normals[index] = glm::vec3(0, 0, 0);
	}
}

__global__ void computeNormal(glm::vec3* pos, glm::vec3* normals, int* triangles, int triangleCount)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index < triangleCount) {
		int v0 = triangles[3*index];
		int v1 = triangles[3*index+1];
		int v2 = triangles[3*index+2];

		glm::vec3 p0 = pos[v0];
		glm::vec3 p1 = pos[v1];
		glm::vec3 p2 = pos[v2];

		glm::vec3 n = glm::normalize(glm::cross(p1-p0, p2-p1));
		atomicAdd(&(normals[v0].x), n.x);
		atomicAdd(&(normals[v0].y), n.y);
		atomicAdd(&(normals[v0].z), n.z);
		atomicAdd(&(normals[v1].x), n.x);
		atomicAdd(&(normals[v1].y), n.y);
		atomicAdd(&(normals[v1].z), n.z);
		atomicAdd(&(normals[v2].x), n.x);
		atomicAdd(&(normals[v2].y), n.y);
		atomicAdd(&(normals[v2].z), n.z);
	}
}

__global__ void resizeNormal(glm::vec3* normals, int dimx, int dimz)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;

	if (i < dimx && j < dimz) {
		int index = j + i * dimz;

		normals[index] = glm::normalize(normals[index]);
	}
}

// Update vbo and nbo
__global__ void sendToVAO(glm::vec3* pos, glm::vec3* normals, float* vbo, float* nbo,
						  int dimx, int dimz)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	
	if (i < dimx && j < dimz) {
		int index = j + i * dimz;

		vbo[4*index  ] = pos[index].x;
		vbo[4*index+1] = pos[index].y;
		vbo[4*index+2] = pos[index].z;
		vbo[4*index+3] = 1;

		nbo[4*index  ] = normals[index].x;
		nbo[4*index+1] = normals[index].y;
		nbo[4*index+2] = normals[index].z;
		nbo[4*index+3] = 0;
	}
}

//Initialize positions and other attributes of vertices
void initCuda(int xdim, int zdim, float dx, float dz, float y0, float mass)
{
	dimx = xdim;
	dimz = zdim;

	vertexCount = dimx*dimz;
	triangleCount = 2*(dimx-1)*(dimz-1);
	edgeCount = 3*(dimx-1)*(dimz-1)+dimx-1+dimz-1;

	cudaMalloc((void**)&pos, vertexCount*sizeof(glm::vec3));
    checkCUDAErrorWithLine("Kernel failed!");
	cudaMalloc((void**)&predicted_pos, vertexCount*sizeof(glm::vec3));
    checkCUDAErrorWithLine("Kernel failed!");
	cudaMalloc((void**)&vel, vertexCount*sizeof(glm::vec3));
    checkCUDAErrorWithLine("Kernel failed!");
	cudaMalloc((void**)&normals, vertexCount*sizeof(glm::vec3));
    checkCUDAErrorWithLine("Kernel failed!");
	cudaMalloc((void**)&lock_pos, vertexCount*sizeof(bool));
    checkCUDAErrorWithLine("Kernel failed!");
	cudaMalloc((void**)&inv_mass, vertexCount*sizeof(float));
    checkCUDAErrorWithLine("Kernel failed!");
	// TEMP
	//cudaMalloc((void**)&acc, vertexCount*sizeof(glm::vec3));
	//checkCUDAErrorWithLine("Kernel failed!");

	cudaMalloc((void**)&triangles, 3*triangleCount*sizeof(int));
    checkCUDAErrorWithLine("Kernel failed!");
	cudaMalloc((void**)&edges, edgeCount*sizeof(Edge));
    checkCUDAErrorWithLine("Kernel failed!");

	threadsPerBlock = dim3(blockSize, blockSize);

	fullBlocksPerGridV = dim3((int)ceil(float(dimx)/float(blockSize)), (int)ceil(float(dimz)/float(blockSize)));
	initVertexAttributes<<<fullBlocksPerGridV, threadsPerBlock>>>(pos, predicted_pos, vel, normals,
		lock_pos, inv_mass, dimx, dimz, dx, dz, y0, mass);
	checkCUDAErrorWithLine("Kernel failed!");

	dim3 fullBlocksPerGridV1 = dim3((int)ceil(float(dimx-1)/float(blockSize)), (int)ceil(float(dimz-1)/float(blockSize)));
	initTriangles<<<fullBlocksPerGridV1, threadsPerBlock>>>(triangles, edges, dimx, dimz);
	checkCUDAErrorWithLine("Kernel failed!");

	threadsPerBlockE = blockSize * blockSize; 
	fullBlocksPerGridE = (int)ceil(float(edgeCount)/float(threadsPerBlockE));

	fullBlocksPerGridT = (int)ceil(float(triangleCount)/float(threadsPerBlockE));

	// calculate stiffness based on solver iterations
	st_stiff = (1 - pow(1-st_stiff, 1.0f/solver_iterations)) * 6;
	b_stiff = 1 - pow(1-b_stiff, 1.0f/solver_iterations);

	cudaMalloc((void**)&fp_constraints, vertexCount*sizeof(FixedPointConstraint));
	checkCUDAErrorWithLine("Kernel failed!");
	cudaMalloc((void**)&st_constraints, edgeCount*sizeof(StretchConstraint));
	checkCUDAErrorWithLine("Kernel failed!");
	cudaMalloc((void**)&b_constraints, edgeCount*sizeof(BendConstraint));
	checkCUDAErrorWithLine("Kernel failed!");

	generateFPConstraints<<<fullBlocksPerGridV, threadsPerBlock>>>(fp_constraints, dimx, dimz,
		dx, dz, y0);
	checkCUDAErrorWithLine("Kernel failed!");
	generateStretchBendConstraints<<<fullBlocksPerGridE, threadsPerBlockE>>>(st_constraints,
		b_constraints, pos, edges, triangles, edgeCount);
	checkCUDAErrorWithLine("Kernel failed!");

	cudaThreadSynchronize();
}

void update(float dt)
{
	preConstraintsUpdate<<<fullBlocksPerGridV, threadsPerBlock>>>(predicted_pos, vel, inv_mass,
		dimx, dimz, gravity, dt);
    checkCUDAErrorWithLine("Kernel failed!");

	// resolve constraints
	resolveFPConstraints<<<fullBlocksPerGridV, threadsPerBlock>>>(fp_constraints, predicted_pos,
		lock_pos, fp_stiff, dimx, dimz);
	checkCUDAErrorWithLine("Kernel failed!");

	for (int i=0; i<solver_iterations; ++i) {
		resolveStretchBendConstraints<<<fullBlocksPerGridE, threadsPerBlockE>>>(st_constraints,
			b_constraints, predicted_pos, inv_mass, lock_pos, st_stiff, b_stiff, edgeCount);
		checkCUDAErrorWithLine("Kernel failed!");
		cudaThreadSynchronize();
	}

	// TEMP
	//resetAcc<<<fullBlocksPerGridV, threadsPerBlock>>>(acc, dimx, dimz);
	//computeAcc<<<fullBlocksPerGridE, threadsPerBlockE>>>(acc, st_constraints, predicted_pos, vel,
	//	inv_mass, Ks, Kd, edgeCount);
	//checkCUDAErrorWithLine("Kernel failed!");
	//applyAcc<<<fullBlocksPerGridV, threadsPerBlock>>>(acc, vel, predicted_pos, lock_pos, dt, dimx, dimz);
	//checkCUDAErrorWithLine("Kernel failed!");

	integrate<<<fullBlocksPerGridV, threadsPerBlock>>>(pos, predicted_pos, vel, dt, dimx, dimz);
	checkCUDAErrorWithLine("Kernel failed!");

	// compute normals
	resetNormal<<<fullBlocksPerGridV, threadsPerBlock>>>(normals, dimx, dimz);
	checkCUDAErrorWithLine("Kernel failed!");
	computeNormal<<<fullBlocksPerGridT, threadsPerBlockE>>>(pos, normals, triangles, triangleCount);
	checkCUDAErrorWithLine("Kernel failed!");
	resizeNormal<<<fullBlocksPerGridV, threadsPerBlock>>>(normals, dimx, dimz);
	checkCUDAErrorWithLine("Kernel failed!");

    cudaThreadSynchronize();
}

void cudaUpdateVAO(float * vbodptr, float * nbodptr)
{
	sendToVAO<<<fullBlocksPerGridV, threadsPerBlock>>>(pos, normals, vbodptr, nbodptr, dimx, dimz);
	checkCUDAErrorWithLine("Kernel failed!");
    cudaThreadSynchronize();
}

void freeCuda() {
	cudaFree(pos);
	cudaFree(predicted_pos);
	cudaFree(vel);
	cudaFree(normals);
	cudaFree(triangles);
	cudaFree(edges);
	cudaFree(lock_pos);
	cudaFree(inv_mass);
	cudaFree(fp_constraints);
	cudaFree(st_constraints);
	cudaFree(b_constraints);
	cudaFree(col_onstraints);
}