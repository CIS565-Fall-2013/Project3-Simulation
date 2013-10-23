CIS565: Project 3: CUDA Simulation and GLSL Visualization
===
Fall 2013
---
Qiong Wang
---

-------------------------------------------------------------------------------
INTRODUCTION
-------------------------------------------------------------------------------
This is a Nbody Simulation Project based on CUDA and OpenGL for CIS 565 GPU Programming. The GLSL visualization gives beautiful results of simulation.

In the first part, the simulation is about how a star system works. There are N small planets moving around the center star with the acceleration computed from the Newton's Gravitation Law. 
The second part is about the general flocking phenomenon in nature. The mechanism of flocking can be listed as the following three main steering behaviors:

1. Alignment: Boids within a certain distance tends to move with the average velocity.
2. Cohesion: Boids within a certain distance tends to move to the average position.
3. Separation: Each boid will give a force to push other close boids away.

-------------------------------------------------------------------------------
FEATURES IMPLEMENTED
-------------------------------------------------------------------------------
###**PART 1**

Basic features:

* Calculating forces between all interacting bodies
* The same, but with shared memory
* Vertex shader code to render a height field
* Fragment shader code to light that height field
* Geometry shader code to create screen facing billboards from rendered points
* Fragment shader code to render those billboards like spheres with simple diffuse shading


Optional features:

* Unique looking for planets
* Runge Kutta integration
* Replace geometry shader billboarding with adding a pyramid pointing in the direction of velocity

###**PART 2**

Features:

* Three steering behavior of boids
* Border handling with velocity bounce
* Fragment shader code to light on pyramid with simple diffuse shading
* Geometry shader code to create screen facing billboards from rendered pyramids pointing in the direction of velocity
* Height Geometry shader of the plane with the height field
* Runge Kutta integration

Features needs to improve:
* Interactive mouse and keyboard is still debugging which will change the view and projection whenenver pressing the left/right, up/down button,
and zoom in and out with the mouse wheel rolling

-------------------------------------------------------------------------------
SCREENSHOTS OF THE FEATURES IMPLEMENTED
-------------------------------------------------------------------------------
* Basic Star System of Part 1

![ScreenShot](https://raw.github.com/GabriellaQiong/Project3-Simulation/master/p1_10170933.PNG)

* Star System of Part 1 with unique looking planets

![ScreenShot](https://raw.github.com/GabriellaQiong/Project3-Simulation/master/p1_10201235.PNG)

* Basic Flocking of Part 2 with pyrimid geometry shader

![ScreenShot](https://raw.github.com/GabriellaQiong/Project3-Simulation/master/p2_10201549.PNG)

* Flocking of Part 2 with diffuse pyrimid geometry shader

![ScreenShot](https://raw.github.com/GabriellaQiong/Project3-Simulation/master/p2_10201550.PNG)

* Flocking of Part 2 with diffuse pyrimid geometry shader and height field

![ScreenShot](https://raw.github.com/GabriellaQiong/Project3-Simulation/master/p2_10201551.PNG)

* Flocking of Part 2 with improved height field

![ScreenShot](https://raw.github.com/GabriellaQiong/Project3-Simulation/master/p2_10221055.PNG)

* Flocking of Part 2 with improved noise height field

![ScreenShot](https://raw.github.com/GabriellaQiong/Project3-Simulation/master/p2_10220903.PNG)

* Flocking of Part 2 with improved ocean wave effect height field

![ScreenShot](https://raw.github.com/GabriellaQiong/Project3-Simulation/master/p2_10221144.PNG)


-------------------------------------------------------------------------------
VIDEOS OF IMPLEMENTATION
-------------------------------------------------------------------------------

Here is the video of the working star system.

[![ScreenShot](https://raw.github.com/GabriellaQiong/Project3-Simulation/master/VideoScreenShot1.PNG)](http://www.youtube.com/watch?v=CIzqrhkuMrI)


This is the video of the flocking process of small pyramids.

[![ScreenShot](https://raw.github.com/GabriellaQiong/Project3-Simulation/master/VideoScreenShot2.PNG)](http://www.youtube.com/watch?v=b5fV_K9yWqA)

Another new video of flocking with ocean wave effect is this:

[![ScreenShot](https://raw.github.com/GabriellaQiong/Project3-Simulation/master/IMG_1985.PNG)](http://www.youtube.com/watch?v=OxMKQQeQXx4)

The youtube links are here if you cannot open the video in the markdown file: http://www.youtube.com/watch?v=CIzqrhkuMrI, http://www.youtube.com/watch?v=b5fV_K9yWqA and http://www.youtube.com/watch?v=OxMKQQeQXx4.

-------------------------------------------------------------------------------
PERFORMANCE EVALUATION
-------------------------------------------------------------------------------
Here is the table for the performance evaluation when changing the number of planets and with shared memory or not for part 1. 

| particle #|with shared memory |  approximate fps  |
|:---------:|:-----------------:|:-----------------:|
|    50     |         no        |       12.77       |
|    64     |         no        |       10.23       |
|   128     |         no        |       5.67        |
|   256     |         no        |       1.28        |
|    50     |        yes        |       10.43       |
|    64     |        yes        |       9.56        |
|   128     |        yes        |       5.72        |
|   256     |        yes        |       1.34        |

Unfortunately, the computers in Moore 100 cannot work when the particle number is bigger than 256, so here only listed four kinds of particle number. Probability, when the particle number is much larger,
the advantage to use shared memory will be apparent. 

-------------------------------------------------------------------------------
REFERENCES
-------------------------------------------------------------------------------
* Newton's Law of Universal Gravitation: http://en.wikipedia.org/wiki/Newton%27s_law_of_universal_gravitation

* Runge Kutta method: http://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods
 
* Usage of Shared memory in GPU: http://http.developer.nvidia.com/GPUGems3/gpugems3_ch31.html

* Flocking (Craig's Paper): http://www.red3d.com/cwr/papers/1999/gdc99steer.pdf

* Russian Roulette rule from Peter and Karl: https://docs.google.com/file/d/0B72qrSEH6CGfbFV0bGxmLVJiUlU/edit

* Noise function: http://stackoverflow.com/questions/4200224/random-noise-functions-for-glsl

-------------------------------------------------------------------------------
ACKNOWLEDGEMENT
-------------------------------------------------------------------------------
Thanks a lot to Patrick and Liam for the preparation of this project. Thank you :)
