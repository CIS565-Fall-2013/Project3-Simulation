CIS565: Project 3: CUDA Simulation and GLSL Visualization
===
Fall 2013
---
---
INTRODUCTION
---
The goal of this project was to leverage the power of GPU in N-body simulations, especially those which require N^2 queries at every step of the simulation.
Inter-body forces usually require calculating contributions from each body on every other body. To this end, there are two parts of this project. 

Part1 simulates gravitational forces between planets, and Part2 simulates flocking-like group behaviors.

PART 1: CUDA NBody Simulation
===

---
FEATURES:
---

 *  Full N-body inter-body gravitational forces
 *  Simulation implemented with a naive and shared memory approach
 *  Vertex shader code to render a height field
 *  Fragment shader code to light that height field
 *  Geometry shader code to create screen facing billboards from rendered points
 *  Fragment shader code to render those billboards like colorful spheres with simple diffuse shading
 *  RK4 (Runge Kutta) and Euler numerical integration methods
 

---
SHARED MEMORY IMPLEMENTATION
---

Before I go into the shared memory approach, I would like to quickly state the naive way to do this,

1. Launch a thread on the GPU for each body bi
2.      In each thread

        totalInteractionForce = 0

        for all N bodies
            calculate force fij from body bj on bi
            totalInteractionForce += fij



PART 2: Your CUDA Simulation
===

To complete this part of the assignment you must implement your own simulation. This can be anything within reason, but two examples that would be well suited are:

* Flocking
* Mass spring cloth/jello

Feel free to code your own unique simulation here, just ask on the Google group if your topic is acceptable and we'll probably say yes.

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
THIRD PARTY CODE POLICY
---
* Use of any third-party code must be approved by asking on our Google group.  
  If it is approved, all students are welcome to use it.  Generally, we approve 
  use of third-party code that is not a core part of the project.  For example, 
  for the ray tracer, we would approve using a third-party library for loading 
  models, but would not approve copying and pasting a CUDA function for doing 
  refraction.
* Third-party code must be credited in README.md.
* Using third-party code without its approval, including using another
  student's code, is an academic integrity violation, and will result in you
  receiving an F for the semester.

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

---
SUBMISSION
---
As with the previous project, you should fork this project and work inside of
your fork. Upon completion, commit your finished project back to your fork, and
make a pull request to the master repository.  You should include a README.md
file in the root directory detailing the following

* A brief description of the project and specific features you implemented
* At least one screenshot of your project running.
* A link to a video of your raytracer running.
* Instructions for building and running your project if they differ from the
  base code.
* A performance writeup as detailed above.
* A list of all third-party code used.
* This Readme file edited as described above in the README section.

---
ACKNOWLEDGEMENTS
---
I adapted the geometry shader code from [this excellent tutorial on the subject](http://ogldev.atspace.co.uk/www/tutorial27/tutorial27.html)
