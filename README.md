CIS565: Project 3: CUDA Simulation and GLSL Visualization
===
Fall 2013
---

![Screenshot](/Renders/PlanetSim.JPG "Solar System")
![Screenshot](/Renders/Boids1.JPG "Flocking Simulation")

Youtube Video of Flocking Simulation:
<dl>
<a href="http://www.youtube.com/watch?v=4-spqjG-MD8" target="_blank"><img src="http://img.youtube.com/vi/4-spqjG-MD8/0.jpg" 
alt="Youtube Video of Simulation Running" width="640" height="480" border="10" /></a>
</dl>

---
NOTE:
---
This project requires an NVIDIA graphics card with CUDA capability! Any card
after the Geforce 8xxx series will work. If you do not have an NVIDIA graphics
card in the machine you are working on, feel free to use any machine in the SIG
Lab or in Moore100 labs. All machines in the SIG Lab and Moore100 are equipped
with CUDA capable NVIDIA graphics cards. If this too proves to be a problem,
please contact Patrick or Liam as soon as possible.

---
INTRODUCTION:
---
This project is divided into two parts. The first folder is a very simple (and not physically accurate) solar system simulation. 
Part two is a GPU Accelerated flocking simulation. Both projects use a combination of CUDA and GLSL OpenGL shaders to do the bulk of the work.


---
CONTENTS:
---
The Project3 root directory contains the following subdirectories:

 *  Planets/ (Part 1: Planet Simulation)
 *  Boids/ (Part 2: Flocking Simulation)


PART 1: CUDA NBody Planet Simulation
===

![Screenshot](/Renders/PlanetSim.JPG "Solar System")

The gravity simulation for this solar system is done in CUDA. 
Each vertex height of the gravity map is also computed with the same CUDA functions.

Once the height map has been generated, it is passed as a VBO to OpenGL using CUDA GL Interop. 
From there, GLSL shaders are used to render the scene.

 *  A Vertex shader code to render a height field
 *  A Fragment shader code to light that height field
 *  A Geometry shader code to create screen facing billboards from planet center points points
 *  Fragment shader code to render those billboards like spheres with simple diffuse shading

In addition to the features mentioned above, a shared memory prefetching technique was used in an attempt to speed up the simulation.
As discussed in the performance section below though, the technique yielded little to no speedup for this sim.

Now we have a beautiful looking (if simple) gravity sim!


PART 2: Flocking Boids
===

The following screenshots consist of 4000 "Boids" obeying simple local rules.
![Screenshot](/Renders/Boids2.JPG "Flocking Simulation")

![Screenshot](/Renders/Boids3.JPG "Flocking Simulation")

The rules current used by the flock are a combination of the following:
1. Attraction (go towards other boids that are far away)
2. Alignment (face the same direction as boids mid distance away)
3. Repulsion (avoid getting too close together)
4. Boundary Conditions (turn around if you get too far from center of frame)
5. Ground Avoidance (pull up!!!)
6. Speed control (Try to maintain a steady cruising speed)

As a little bonus, press 'b' while running the sim.
To adjust any of the rules, the quickest way is to alter the "WorldProps" struct in main.cpp

The simulation spits out a position and velocity for each BOID. I then use a geometry shader to create each boid's geometry.


---
NOTES ON GLM:
---
This project uses GLM, the GL Math library, for linear algebra. You need to
know two important points on how GLM is used in this project:

* In this project, indices in GLM vectors (such as vec3, vec4), are accessed
  via swizzling. So, instead of v[0], v.x is used, and instead of v[1], v.y is
  used, and so on and so forth.
* GLM Matrix operations work fine on NVIDIA Fermi cards and later, but
  pre-Fermi cards do not play nice with GLM matrices. As such, in this project,
  GLM matrices are replaced with a custom matrix struct, called a cudaMat4, found
  in cudaMat4.h. A custom function for multiplying glm::vec4s and cudaMat4s is
  provided as multiplyMV() in intersections.h.


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


---
ACKNOWLEDGEMENTS
---
I adapted the geometry shader code from [this excellent tutorial on the subject](http://ogldev.atspace.co.uk/www/tutorial27/tutorial27.html)
