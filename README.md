
---
INTRODUCTION:
---
In the part 1 of this project ,the following features have ben implemented :

 *  Calculating forces between all interacting bodies
 *  The same, but with shared memory
 *  Vertex shader code to render a height field
 *  Fragment shader code to light that height field
 *  Geometry shader code to create screen facing billboards from rendered points
 *  Fragment shader code to render those billboards like spheres with simple diffuse shading

In the Part 2 of this Project, a flocking simulation has been created. In order for the flocking
to work, it is a combination of Allignment, seperation and cohesive velocities. 
The above 3 velocites are calculated based upon the position and velocities of each object with
every other object within a confined neighbourhood radius.

---
Video :
---
[Flocking Video](http://www.youtube.com/watch?v=QnTGIj2jIUw)

---
Screen Shots:
---
Planetary Simulation 
![alt tag](https://raw.github.com/vivreddy/Project3-Simulation/master/images/planets.jpg)

Flocking Simulation
![alt tag](https://raw.github.com/vivreddy/Project3-Simulation/master/images/flocking.jpg)

---
PERFORMANCE EVALUATION
---
The performance evaluation for the simulation was done by comparing fps while using
global memory and shared memory.

![alt tag](https://raw.github.com/vivreddy/Project3-Simulation/master/images/table.JPG)

Here intially when the number of bodies are less, the contribution of shared memory for the 
increase in fps is not so significant. But once the number of bodies are increased more than 
1000 we need a significant contribution of using shared memory over global memory




