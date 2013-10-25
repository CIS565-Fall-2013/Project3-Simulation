CIS565: Project 3: CUDA Simulation and GLSL Visualization
===
Fall 2013

PART 1: CUDA NBody Simulation
===

The first part of this project involves a N-body simulation of planets revolving
the sun. Each planet is assigned a random velocity. At every frame, acceleration 
is calculated using the law of universal gravitation. 

Each planet is represented as a sphere lit by the star at position (0,0,0).

![alt text](./Images/planets.jpg "Planets")


Basic features include:

* Force calculation between all bodies using global and shared memory
* Height field rendering
* Fragment shader rendering of spherical planets

PART 2: Flocking Simulation
===
For part two, I implemented a flocking simulation using the methods described [here](http://www.red3d.com/cwr/boids/).
Each boid is affected by the following:

* Cohesion -- boid steers to move to the center of mass of all boids in its neighborhood
* Alignmnet -- boid heads toward the average direction of its neighbors
* Separation -- boid steers to avoid being too close to its neighbors
* Avoidance -- boid tries to avoid a boundary. In my implementation, I created an imaginary 
	cube to constrain the boids.
* Wander -- adds a random velocity (constrained to a unit sphere) to each boid to create
	interesting movement

Here are some results:
![alt text](./Images/1000Boids.jpg "1000 Boids")

![alt text](./Images/5000Boids.jpg "5000 Boids")

![alt text](./Images/7000Boids.jpg "7000 Boids")

[Here](https://vimeo.com/77762701) is a video of the simulation. There are 5000 boids in the scene

---
PERFORMANCE EVALUATION
---
The performance evaluation is where you will investigate how to make your CUDA
programs more efficient using the skills you've learned in class. You must
perform at least one experiment on your code to investigate the positive or
negative effects on performance. 

For this Project, one of these experiments should be a comparison between the 
global and shared memory versions of the acceleration calculation function at
varying block sizes.

A good metric to track would be number of frames per second, 
or number of objects displayable at 60fps.

We encourage you to get creative with your tweaks. Consider places in your code
that could be considered bottlenecks and try to improve them. 

Each student should provide no more than a one page summary of their
optimizations along with tables and or graphs to visually explain any
performance differences.
