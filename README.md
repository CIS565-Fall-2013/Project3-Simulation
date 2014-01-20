CUDA Simulation and GLSL Visualization
===
Fall 2013
---
INTRODUCTION:
---

This project is divided into two parts. Part one creates a 3D visualization of an N-Body system 
simulated using CUDA and OpenGL shaders. Part two is a flocking simulation with pyramid boids colored using velocities.

-------------
PART 1: CUDA NBody Simulation
===

![Alt text](Part1/project running.jpg?raw=true)

This part of the project creates a NBody simulation on 2-D plane with gravitational field visualized as height field.

The followings are basic features:

 *  Calculating forces between all interacting bodies
 *  The same, but with shared memory
 *  Vertex shader code to render a height field
 *  Fragment shader code to light that height field
 *  Geometry shader code to create screen facing billboards from rendered points
 *  Fragment shader code to render those billboards like spheres with simple diffuse shading

Besides, Runge Kutta integration is also implemented to make simulation more stable

Project video:https://www.youtube.com/watch?v=RwWNOOuEdHM

-------------
PART 2: CUDA Flocking Simulation
===
![Alt text](Part2/project running.jpg?raw=true)
![Alt text](Part2/project running1.jpg?raw=true)
![Alt text](Part2/project running2.jpg?raw=true)
![Alt text](Part2/project running3.jpg?raw=true)
![Alt text](Part2/project running4.jpg?raw=true)

This part of the project simulates flocking behavior described by Craig Reynolds http://www.red3d.com/cwr/boids/ using the combination of 4 behaviors:

* Alignment: the boid tries to maintain a velocity close the average velocity of its neighbors
* Cohesion: the boid steers itself to the center of mass of its neighbors
* Separation: the boid maintains a proper distance with its neighbor to avoid crowding
* Seek: the boid tends to move towards a target that can be moved by mouse motion


Features:

* 3d flocking simulation with tens of thousands of boids on GPU
* Pyramid shaped boids generated in geometry shader pointing at direction of velocity
* Velocity-encoded color with diffuse shading
* Mouse-controlled flock

Project video:https://www.youtube.com/watch?v=eWeW96t6cE0



---
PERFORMANCE EVALUATION
---
For the NBody simulation part, shared memory access only shows its advantages when simulated body count is large. This could be due to my graphic card has more bandwidth than an average card, with 384-bit memory interface. So, if N is small, access to global memory is not so much of a burden.

![Alt text](Part1/Performance evaluation.jpg?raw=true)



---
ACKNOWLEDGEMENTS
---
I adapted the geometry shader code from [this excellent tutorial on the subject](http://ogldev.atspace.co.uk/www/tutorial27/tutorial27.html)
