CIS565: Project 3: CUDA Simulation and GLSL Visualization
===
Fall 2013

PART 1: CUDA NBody Simulation
===

The first part of this project involves a N-body simulation of planets revolving
the sun. Each planet is assigned a random velocity. At every frame, acceleration 
is calculated using the law of universal gravitation. 

Each planet is represented as a sphere lit by the star at position (0,0,0).

[!alt text](./Images/planets.jpg "Planets")


Basic features include:
*Force calculation between all bodies using global and shared memory
*Height field rendering
*Fragment shader rendering of spherical planets

PART 2: Flocking Simulation
===


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
Resources
---
*Referred to the CIS462/562 behavior animation notes

---
SELF-GRADING
---
* On the submission date, email your grade, on a scale of 0 to 100, to Liam,
  liamboone+cis565@gmail.com, with a one paragraph explanation.  Be concise and
  realistic.  Recall that we reserve 30 points as a sanity check to adjust your
  grade.  Your actual grade will be (0.7 * your grade) + (0.3 * our grade).  We
  hope to only use this in extreme cases when your grade does not realistically
  reflect your work - it is either too high or too low.  In most cases, we plan
  to give you the exact grade you suggest.
* For late assignments there will be a 50% penaly per week.
* Projects are not weighted evenly, e.g., Project 0 doesn't count as much as
  the path tracer.  We will determine the weighting at the end of the semester
  based on the size of each project.
