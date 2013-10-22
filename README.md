CIS565: Project 3: CUDA Simulation and GLSL Visualization
===
Fall 2013
---
#Part 1 : NBody Simulator

###Overview

We have implemented a very straight forward nbody simulator that simulates a simple planetary system with a singular star that resides at the world's center (0,0,0).

###Performance Analysis

We have implemented both a naive accumulator, which uses global memory, and a naive shared memory accumulator, which iteratively brings the data into shared memory.
We see that while shared memory has performance benefits over using global memory at smaller block sizes, this does not hold true as we increase block size. 
It would probably be better to change the structure around and apply reduce sum instead of an iterative sum to accumulate the forces from all of the planets/stars in the system.

100 Bodies, dt = 0.5

Naive Accumulator (Global Memory)

Block Size | fps
---|---
32 | 10.93
64 | 19.25
128 | 19.96


Shared Memory 

Block Size | fps
----|----
32 | 17.93
64 | 21.19
128 | 19.92

------
###Video

https://vimeo.com/77479666

---
#Part 2 : Flocking Simulator

###Overview

As part 2 of this project, we chose to write a simple flocking simulator based on Reynold's paper on flocking behavior.  We have implemented
the 3 simple rules of flocking along with simple flee steeering to avoid the invisible walls in the cube that we have set up.

###Features

We have implemented the following features of flocking:
1.  Alignment
2.  Cohesion
3.  Separation
4.  Wall Avoidance

In addition to this we have changed the shaders to allow for the boids to be drawn as triangles, where the triangle is drawn in the direction of velocity of the boid.
This was done by creating another VBO that was mapped.  This VBO copied velocities on every iteration of the CUDA call and sent it as another vertex attribute.
The vertex shader took both the position and the velocity and output the position and velocity, and the geometry shader took the position and velocity as single element
arrays.  The "right" and "left" directions for the triangle are determined by the cross product of the normalized velocity and the world's UP direction (0,0,1). If the boid is 
moving upwards, the "right" direction is determined by the cross product of the normalized velocity and the normalized direction to the camera. 

Finally we have added motion blur by using GL's accumulation buffer.

###Performance Analysis

We started the simulator with 25 boids on a block size of 128, rendering them as triangles, where the tip is pointing in the direction of velocity.  We found that this rendered at about 58 fps.  
When increasing to 2500 boids, the simulator still runs at around ~55 fps, dipping to ~40 fps when other functions on the computer are taking away from resources.
Since this is O(n^2) function that we are performing, and we are using shared memory, the GPU seems to handle this well. 

We looked at the effects of changing block size on the performance at 2500 boids.  Since all the accumulators that we implemented used shared memory, 
theoretically, the size of shared memory would affect how many times we need to copy from global into shared memory and how much shared memory we can access at 
a single moment.

Block Size | fps
--- | ---
1 | 1.6
4 | 11.9
8 | 23
16 | 36-40
32 | 51
64 | 43
128 | 45

Interestingly, on each run of the simulator, we noticed that it would take a moment or two to ramp up.  Once it ramped up to around 58-60 fps, it would gradually drop until it stabilized.
All the meaurements taken above are taken when the simualtor stabilized.

###Video

https://vimeo.com/77476551

---
Acknowledgements

Many thanks to Ishaan for his help with transforming the shaders to draw the boids as they currently are. 

Much of my implementation of the flocking simulator was informed by the following:
[Game Dev Tutorial on 2D Flocking]:http://gamedev.tutsplus.com/tutorials/implementation/the-three-simple-rules-of-flocking-behaviors-alignment-cohesion-and-separation/
[Reynolds' Flocking Paper]:www.­red3d.­com/­cwr/­boids/­
