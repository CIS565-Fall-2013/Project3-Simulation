CIS565: Project 3: CUDA Simulation and GLSL Visualization
===
Fall 2013
---
Due Tuesday, 10/22/2013 by 11:59:59 pm
---

PART 1: CUDA NBody Simulation
===
---
REQUIREMENTS:
---
I was provided with code for:
 *  Initialization
 *  Rendering to the screen
 *  Some helpful math functions
 *  CUDA/OpenGL inter-op

I implemented the following features:
 *  Calculating forces between all interacting bodies
 *  The same, but with shared memory (In main.cpp there is a compile flag #define VISUALIZE that can be set to 0 in order to only run the simulation portion of the code and not the costly visualizer)
 *  Vertex shader code to render a height field
 *  Fragment shader code to light that height field
 *  Geometry shader code to create screen facing billboards from rendered points
 *  Fragment shader code to render those billboards like spheres with simple diffuse shading

PART 2: Flocking
===

I then implemented a flocking simulation, building off of the base code from Part 1.

![Flocking simulation](flocking.PNG)

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
