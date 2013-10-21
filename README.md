-------------------------------------------------------------------------------
CIS565 Project 1: CUDA Raytracer
-------------------------------------------------------------------------------
Ricky Arietta Fall 2013
-------------------------------------------------------------------------------
-------------------------------------------------------------------------------

---
INTRODUCTION:
---
In this project you will be creating a 3D visualization of an N-Body system 
simulated using CUDA and OpenGL shaders. You will also be creating your own 
simulation of choice.

---
PART 1: CUDA NBody Simulation
===
For this part of the simulation, I implemented the height field and planet shaders
to convert the simple point and plane rendering to a physical system that
included a simple light model for the planets and a demonstration of the gravitational
field within the texture mapped plane. Here is a shot of the program running.

![NBody Simulation] (https://raw.github.com/rarietta/Project3-Simulation/master/readme_images/NBody.bmp)

In order to correctly update the motion of the planets, I implemented an
acceleration computation. This function took in a planet's own position
and another planet's position and calculated the gravitational force in the
direction of the other planet, using the universal planetMass to then
derive the acceleration along that vector. You can see from the above image
that this calculation was correct for each planet, as demonstrated by the consistent
motion and gravitational fields in the plane.

Furthermore, this iterative acceleration integration was done both naively,
wherein each thread loaded and computed the force from each other planet, and
more efficiently using shared memory. In the shared memory approach, the
planets were split up into blocks, and in each block a single thread
would load its own data into shared memory on the device. From there, all
threads in the block would calculate the force accumulated from each planet
in this tile, and continue on to the next until accumulating force from
all planets. Then the velocities and positions for the timestep would
be updated.


PART 2: Your CUDA Simulation
===

For my CUDA simulation in Part 2, I implemented a cloth sim. This simulation treats
cloth as a uniform grid of particles. Therefore, my first adaptation from the given
code was to modify the planet position generation to instead generate a uniform grid
of size N*N. Here is an example of that basic grid.

![Uniform Grid of Particles] (https://raw.github.com/rarietta/Project3-Simulation/master/readme_images/uniform_grid.bmp)

After establishing the grid of particles, I had to implement a system of springs,
which I stored as an array of glm::vec3 objects. Each particle indexed to NUM_NEIGHBORS=12
unique springs in the spring array, from indices [index*NUM_NEIGHBORS]
through [index*NUM_NEIGHBORS + 11] (index, of course, being the index of the particle in question
from [0,N^2). The 12 springs represent the 3 sets of cloth-particle relationships
illustrated in the picture below.

![Relationship of each cloth particle with its 12 neighbors] (https://raw.github.com/rarietta/Project3-Simulation/master/readme_images/cloth_springs.bmp)

Each spring vector thus contained 3 elements: the index of the opposite particle, the
spring constant "k" as defined by Hooke's law, which was used to calculate forces
between particles, and the resting length L0 of the spring as defined by the distance
in the uniform grid.

After calculating and storing all these relationships, the forces at each time step could be
computed just as in Part 1. To demo the simulation, I have created an implicitly defined (though
invisible for our purposes) sphere located at the origin, and introduced a simple gravitational
force into the scene. This video represents the results of dropping the cloth
onto the sphere.

[![High-Res Cloth Video](https://raw.github.com/rarietta/Project3-Simulation/master/readme_images/video_shot_1.bmp)] (https://raw.github.com/rarietta/Project3-Simulation/master/readme_images/Project3%202013-10-20%2019-50-54-450.avi)

Here is the same simulation run at a lower cloth resolution.

[![Low-Res Cloth Video](https://raw.github.com/rarietta/Project3-Simulation/master/readme_images/video_shot_2.bmp)] (https://raw.github.com/rarietta/Project3-Simulation/master/readme_images/Project3%202013-10-20%2020-05-10-283.avi)

And here are some screenshots of the simulation in higher quality. Note in these, the
cloth is not as offset as it was in the video simulation, so it wraps around the
sphere more uniformly when it lands.

![Cloth on sphere 1] (https://raw.github.com/rarietta/Project3-Simulation/master/readme_images/cloth_on_ball_1.bmp)
![Cloth on sphere 2] (https://raw.github.com/rarietta/Project3-Simulation/master/readme_images/cloth_on_ball_2.bmp)
![Cloth on sphere 3] (https://raw.github.com/rarietta/Project3-Simulation/master/readme_images/cloth_on_ball_3.bmp)
![Cloth on sphere 4] (https://raw.github.com/rarietta/Project3-Simulation/master/readme_images/cloth_on_ball_4.bmp)

---
PERFORMANCE EVALUATION
===

For my performance analysis, I conducted a comparison between the 
global and shared memory versions of the acceleration calculation function at
varying block sizes.

Some information: I am running my program on an NVidia GEForce 730m Graphics Card with
4000 MB of RAM.

These results indicate that there is actually not much performance increase in the
shared memory implementation. The difference in Frame Rate shown below between the two
versions is almost nonexistent.

![Frame Rates 1] (https://raw.github.com/rarietta/Project3-Simulation/master/readme_images/performance_chart.bmp)
![Frame Rates 2] (https://raw.github.com/rarietta/Project3-Simulation/master/readme_images/performance_graph.bmp)


