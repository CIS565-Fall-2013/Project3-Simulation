#include <stdio.h>
#include <cuda.h>
#include <cmath>
#include "glm/glm.hpp"
#include "utilities.h"
#include "kernel.h"


//GLOBALS
dim3 threadsPerBlock(blockSize);

int numObjects;

BoidProps * dev_boids;
glm::vec4 * dev_netForces;


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
		system("pause");
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

//Generate randomized flock (in range (-mapDims.x/2:mapDims.x/2, -mapDims.y/2:mapDims.y/2, 0:mapDims.z))
__global__
	void generateRandomFlock(int time, int N, BoidProps * boids, WorldProps properties)
{
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	if(index < N)
	{

		glm::vec3 randPos = properties.InitialDims*(generateRandomNumberFromThread(time, index)-glm::vec3(0.5f,0.5f,0.0f));
		boids[index].pos.x = randPos.x;
		boids[index].pos.y = randPos.y;
		boids[index].pos.z = randPos.z;

		glm::vec3 velocity = properties.InitialMaxVel*generateRandomNumberFromThread(time+1, index);
		float speed = glm::length(velocity);
		boids[index].speed = speed;
		if(speed > properties.MinSpeed)
		{
			boids[index].heading = velocity/speed;
		}else{
			boids[index].heading = glm::vec3(1,0,0);
			boids[index].speed = properties.MinSpeed;
		}

		boids[index].rollAngle = 0.0;
	}
}



__device__ glm::vec4 ruleStayInBounds(const WorldProps world, const BoidProps me)
{
	//Normalize
	glm::vec4 forceIntensity(0);
	
	//X Direction
	if(me.pos.x < -world.WorldBounds.x)
		forceIntensity.x = world.WallStiffness*(world.WorldBounds.x-me.pos.x);
	else if(me.pos.x > world.WorldBounds.x)
		forceIntensity.x = world.WallStiffness*(world.WorldBounds.x-me.pos.x);

	//Y Direction
	if(me.pos.y < -world.WorldBounds.y)
		forceIntensity.y = world.WallStiffness*(world.WorldBounds.y-me.pos.y);
	else if(me.pos.y > world.WorldBounds.y)
		forceIntensity.y = world.WallStiffness*(world.WorldBounds.y-me.pos.y);

	
	//Z Direction
	if(me.pos.z < 0)//Wall at Z == 0, not -WorldBounds.z
		forceIntensity.z = world.WallStiffness*(world.WorldBounds.z-me.pos.z);
	else if(me.pos.z > world.WorldBounds.z)
		forceIntensity.z = world.WallStiffness*(world.WorldBounds.z-me.pos.z);

	return forceIntensity;
	
}

__device__ glm::vec4 ruleAvoidGround(const WorldProps world, const BoidProps me)
{
	//Normalize
	float forceIntensity = (world.GroundAvoidanceHeight-me.pos.z)/world.GroundAvoidanceHeight;

	if(forceIntensity < 0)
		forceIntensity = 0;

	return glm::vec4(0,0,forceIntensity*world.GroundAvoidanceForce,0);
	
}

__device__ glm::vec4 ruleDoABarrelRoll(const WorldProps world, const BoidProps me)
{	
	return glm::vec4(0,0,0,world.BarrelRoll);	
}

__device__ glm::vec4 ruleSpeedControl(const WorldProps world, const BoidProps me)
{	
	return glm::vec4(world.SpeedControlForce*(world.TargetSpeed-me.speed)*me.heading,0.0);	
}


__device__ 
	glm::vec4 applyIndividualRules(const WorldProps world, const BoidProps me)
{
	//TODO: Apply rules
	glm::vec4 netForce = glm::vec4(0);

	netForce += ruleAvoidGround(world, me);
	netForce += ruleStayInBounds(world, me);
	netForce += ruleDoABarrelRoll(world, me);
	netForce += ruleSpeedControl(world, me);

	return netForce;
}

__device__ glm::vec4 ruleAttraction(const WorldProps world, const BoidProps me, const BoidProps them, const float radius, const glm::vec3 towardsThem)
{
	//Constant force in zone
	return glm::vec4(towardsThem*world.AttractionZone.z, 0.0f);
}

__device__ glm::vec4 ruleAlignment(const WorldProps world, const BoidProps me, const BoidProps them, const float radius, const glm::vec3 towardsThem)
{
	
	return glm::vec4(world.AlignmentZone.z*(them.heading-me.heading), 0.0f);
}


__device__ glm::vec4 ruleRepulsion(const WorldProps world, const BoidProps me, const BoidProps them, const float radius, const glm::vec3 towardsThem)
{
	//Constant force in zone
	return glm::vec4(-towardsThem*world.RepulsionZone.z, 0.0f);
}

__device__ 
	glm::vec4 applyPairwiseRules(const WorldProps world, const BoidProps me, const BoidProps them)
{
	glm::vec4 netForce = glm::vec4(0);

	glm::vec3 dist = them.pos - me.pos;

	float radius = glm::length(dist);
	glm::vec3 towardsThem = dist/radius;
	float distDot = glm::dot(towardsThem,me.heading); 

	if(distDot > world.ViewAngleCos)
	{
		//Only apply these rules if we can see them
		
		//Check ranges
		if(/*min*/world.AttractionZone.x < radius && /*max*/radius < world.AttractionZone.y);
			netForce += ruleAttraction(world, me, them, radius, towardsThem);
			
		if(/*min*/world.AlignmentZone.x < radius && /*max*/radius < world.AlignmentZone.y);
			netForce += ruleAlignment(world, me, them, radius, towardsThem);
			
		if(/*min*/world.RepulsionZone.x < radius && /*max*/radius < world.RepulsionZone.y);
			netForce += ruleRepulsion(world, me, them, radius, towardsThem);
	}
	return netForce;
}


__device__
	glm::vec4 computeNetForce(const int N, const WorldProps world, const int myIndex, const BoidProps me, const BoidProps* boids)
{
	extern __shared__ BoidProps shBoids[]; 
	glm::vec4 netForce = glm::vec4(0,0,0,0);//Use 4th field for roll torque

	//Compute for each tile
	int numTiles = N/blockDim.x;
	if(N % blockDim.x != 0)
		numTiles++;//Add a tile for the extras

	//For each full tile
	for(int tile = 0; tile < numTiles; ++tile){

		int tileOffset = tile*blockDim.x;
		//Load into shared memory using coallesed acces
		int loadIndex = threadIdx.x + tileOffset;
		if(loadIndex < N)
			shBoids[threadIdx.x] = boids[loadIndex];

		//Wait for load to finish
		__syncthreads();

		//Perform update for entire tile using shared memory
		//No bank conflicts because this is broadcast
		for(int i = 0; i < blockDim.x; ++i)
		{
			int idx = tileOffset+i;
			if(idx < N && idx != myIndex)//Don't process 
				netForce += applyPairwiseRules(world, me, shBoids[i]);
			else
				break;
		}

		__syncthreads();

	}

	netForce += applyIndividualRules(world, me);
	return netForce;
}

//Simple Euler integration scheme. Update desired velocities
__global__
	void updateForces(int N, WorldProps world, float dt, BoidProps* boids, glm::vec4* netForces)
{ 

	int index = threadIdx.x + (blockIdx.x * blockDim.x);

	if(index < N) 
	{
		netForces[index] = computeNetForce(N, world, index, boids[index], boids);
	}
}

__global__
	void updateS(int N, WorldProps world, float dt, BoidProps* boids, glm::vec4 * netForces)
{
	int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if( index < N )
	{
		//TODO: Apply forces to Boids
		//TODO: Encorporate mass
		//TODO: Include gravity component
		//TODO: Include non-holonomic controller
		glm::vec4 netF = netForces[index];

		//Simulated force saturation
		float FMagnitude = glm::sqrt(netF.x*netF.x+netF.y*netF.y+netF.z*netF.z);
		if(FMagnitude > world.MaxForceMagnitude)
		{
			netF *= (world.MaxForceMagnitude/FMagnitude);
		}

		glm::vec3 delV = glm::vec3(netF)/1.0f*dt;//a = F/m

		glm::vec3 vel = glm::vec3(boids[index].heading)*boids[index].speed;
		vel += delV;
		float mag = glm::length(vel);
		if(mag > world.MinSpeed){
			boids[index].heading = vel/mag;//Only change direction if we are moving, otherwise maintain heading
		}else{
			mag = world.MinSpeed;
		}
		boids[index].speed = mag;
		
		//Change roll 
		boids[index].rollAngle += netF.w*dt;

		boids[index].pos += boids[index].heading*boids[index].speed*dt;
	}
}

//Update the vertex buffer object
//(The VBO is where OpenGL looks for the positions for the planets)
__global__
	void sendToVBO(int N, BoidProps * boids, float * vbo)
{
	int index = threadIdx.x + (blockIdx.x * blockDim.x);

	if(index<N)
	{

		BoidProps me = boids[index];
		//Position
		vbo[boidVBO_PositionOffset + boidVBOStride*index + 0] =  me.pos.x;
		vbo[boidVBO_PositionOffset + boidVBOStride*index + 1] =  me.pos.y;
		vbo[boidVBO_PositionOffset + boidVBOStride*index + 2] =  me.pos.z;
		vbo[boidVBO_PositionOffset + boidVBOStride*index + 3] =  1.0f;

		//TODO: Allow Roll
		
		glm::vec3 Forward =  me.heading;

		glm::vec3 Right = glm::cross(Forward, glm::vec3(0,0,1));

		if(glm::length(Right) == 0)
		{
			Right = glm::vec3(1,0,0);//Pointing straight up. This rare edge case is okay to handle in odd ways.
		}
		//Up
		glm::vec3 Up = glm::cross(Right, Forward);

		glm::vec4 RollUp = glm::rotate(glm::mat4(1.0f), me.rollAngle, Forward)*glm::vec4(Up,0.0);

		vbo[boidVBO_UpOffset + boidVBOStride*index + 0] = RollUp.x;
		vbo[boidVBO_UpOffset + boidVBOStride*index + 1] = RollUp.y;
		vbo[boidVBO_UpOffset + boidVBOStride*index + 2] = RollUp.z;

		//Forward
		vbo[boidVBO_ForwardOffset + boidVBOStride*index + 0] = Forward.x;
		vbo[boidVBO_ForwardOffset + boidVBOStride*index + 1] = Forward.y;
		vbo[boidVBO_ForwardOffset + boidVBOStride*index + 2] = Forward.z;

		//Color
		vbo[boidVBO_ColorOffset + boidVBOStride*index + 0] = 1.0f;
		vbo[boidVBO_ColorOffset + boidVBOStride*index + 1] = 1.0f;
		vbo[boidVBO_ColorOffset + boidVBOStride*index + 2] = 1.0f;

		//Shape
		vbo[boidVBO_ShapeOffset + boidVBOStride*index + 0] = 0.5f;//Length
		vbo[boidVBO_ShapeOffset + boidVBOStride*index + 1] = 0.5f;//Wingspan
		vbo[boidVBO_ShapeOffset + boidVBOStride*index + 2] = 0.25f;//Delta
		vbo[boidVBO_ShapeOffset + boidVBOStride*index + 3] = -30.0f;//Wing Deflection (degrees)
	}
}


/*************************************
* Wrappers for the __global__ calls *
*************************************/

//Initialize memory, update some globals
void initCuda(int N, WorldProps properties)
{
	numObjects = N;
	dim3 fullBlocksPerGrid((int)ceil(float(N)/float(blockSize)));

	//Initialize status arrays
	cudaMalloc((void**)&dev_boids, N*sizeof(BoidProps));
	checkCUDAErrorWithLine("Kernel failed Init()!");
	cudaMalloc((void**)&dev_netForces, N*sizeof(glm::vec4));
	checkCUDAErrorWithLine("Kernel failed Init()!");

	generateRandomFlock<<<fullBlocksPerGrid, blockSize>>>(1, numObjects, dev_boids, properties);
	checkCUDAErrorWithLine("Kernel failed!");
	cudaThreadSynchronize();
}

void cudaNBodyUpdateWrapper(float dt, WorldProps worldProps)
{
	dim3 fullBlocksPerGrid((int)ceil(float(numObjects)/float(blockSize)));
	updateForces<<<fullBlocksPerGrid, blockSize, blockSize*sizeof(BoidProps)>>>(numObjects, worldProps, dt, dev_boids, dev_netForces);
	checkCUDAErrorWithLine("Kernel Update failed!");
	updateS<<<fullBlocksPerGrid, blockSize>>>(numObjects, worldProps, dt, dev_boids, dev_netForces);
	checkCUDAErrorWithLine("Kernel failed!");
	cudaThreadSynchronize();
}

void cudaUpdateVBO(float * vbodptr)
{
	dim3 fullBlocksPerGrid((int)ceil(float(numObjects)/float(blockSize)));
	sendToVBO<<<fullBlocksPerGrid, blockSize>>>(numObjects, dev_boids, vbodptr);
	cudaThreadSynchronize();
}


