CIS565: Project 3: CUDA Simulation and GLSL Visualization
===
Fall 2013
---
INTRODUCTION:
---
In this project you will be creating a 3D visualization of an N-Body system 
simulated using CUDA and OpenGL shaders. You will also be creating your own 
simulation of choice.

This project is divided into two parts. Part one will consist mostly of a 
tutorial style walkthrough of creating the N-Body sim. Part two is an open 
ended assignment to create your own simulation. This simulation can be virtually 
anything you choose, but will require approval for ideas not listed in this 
readme. 

You are also free to do as many extra simulations as you like!

---
CONTENTS:
---
The Project3 root directory contains the following subdirectories:

 *  Part1/
     *  resources/ the screenshots used in this readme.
     *  src/ contains the provided code. __NOTE:__ Shader code will be located in the PROJ3_XYZ folders
     *  PROJ_WIN/ contains a Visual Studio 2010 project file with different configurations
         *  Debug (v4.0)
         *  Release (v4.0)
         *  Debug (v5.5)
         *  Release (v5.5)
     *  PROJ_NIX/ contains a Linux makefile for building and running on Ubuntu 
        12.04 LTS. Note that you will need to set the following environment
        variables (you may set these any way that you like. I added them to my .bashrc): 
         *  PATH=$PATH:/usr/local/cuda-5.5/bin
         *  LD_LIBRARY_PATH=/usr/local/cuda-5.5/lib64:/lib
 *  Part2/ you will fill this with your own simulation code.

__NOTE:__ Since I do not use Apple products regularly enough to know what I'm doing I did not create a Mac friendly version of the project. I will award a +5 point bounty to the first person to open a pull request containing an OSX compatible version of the starter code. All runners up will receive +100 awesome points.

PART 1: CUDA NBody Simulation
===

---
REQUIREMENTS:
---
In this project, you are given code for: 
 *  Initialization
 *  Rendering to the screen
 *  Some helpful math functions
 *  CUDA/OpenGL inter-op

You will need to implement the following features:
 *  Calculating forces between all interacting bodies
 *  The same, but with shared memory
 *  Vertex shader code to render a height field
 *  Fragment shader code to light that height field
 *  Geometry shader code to create screen facing billboards from rendered points
 *  Fragment shader code to render those billboards like spheres with simple diffuse shading

You are NOT required to implement any of the following features:
 *  Prefetching (__NOTE:__ to receive +5 for this feature it must be discussed in your performance section)
 *  Tessellation shader code to refine the heightfield mesh in regions of interest
 *  Render the height map as a quad and use parallax occlusion mapping in the fragment shader to simulate the height field
 *  More interesting rendering of the scene (making some planets light sources would be cool, or perhaps adding orbit trails)
 *  Textures for the planets and/or unique looking planets
 *  Replace geometry shader billboarding with adding in simple models (e.g. a pyramid pointing in the direction of velocity)
 *  Collisions
 *  Runge Kutta integration (or anything better than the Euler integration provided)

Since we had some problems going live on time with this project you can give yourself a +5 point boost for including up to two of the above extra features. For example, adding collisions and textured planets along with completing all other required components can get you a 110% score.

---
WALKTHROUGH
---



PART 2: Your CUDA Simulation
===

To complete this part of the assignment you must implement your own simulation. This can be anything within reason, but two examples that would be well suited are:

* Flocking
* Mass spring cloth/jello

Feel free to code your own unique simulation here, just ask on the Google group if your topic is acceptable and we'll probably say yes.

---
README
---
All students must replace the contents of this Readme.md in a clear manner with 
the following:

* A brief description of the project and the specific features you implemented.
* At least one screenshot of your project running.
* A 30 second or longer video of your project running.  To create the video you
  can use http://www.microsoft.com/expression/products/Encoder4_Overview.aspx 
* A performance evaluation (described in detail below).

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
