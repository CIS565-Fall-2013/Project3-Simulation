#ifndef BOID_H
#define BOID_H

const __device__ float radius = 5.0;
const __device__ float Mass = 1.0;
const __device__ float Inertia = 1.0;
const __device__ float MaxVelocity = 5.0;
const __device__ float MaxForce = 4.0;
const __device__ float MaxTorque = 2.0;
const __device__ float MaxAngVel = 2.0;
const __device__ float Kv0 = 1.0;
const __device__ float Kp1 = 1.0;
const __device__ float Kv1 = 1.0;
const __device__ float KArrival = 0.1;
const __device__ float KDeparture = 10.0;
const __device__ float KNoise = 10.0;
const __device__ float KWander = 6.0;
const __device__ float KAvoid = 1.0;
const __device__ float TAvoid = 1.0;
const __device__ float RNeighborhood = 10000.0;
const __device__ float KSeparate = 20.0;
const __device__ float KAlign = 5.0;
const __device__ float KCohesion = 1.0;

const __device__ float M_PI = 3.14159265358979323846f;		// per CRC handbook, 14th. ed.
const __device__ float M_PI_2 = 1.570796326794897f;				// PI/2
const __device__ float M2_PI = 6.283185307179586f;			// PI*2


enum Behavior 
{
	Seek,
	Flee,
	Arrival,
	Departure,
	Alignment,
	Cohesion,
	Separation,
	Flocking
};

#endif