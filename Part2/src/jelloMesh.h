#ifndef JELLO_MESH_H
#define JELLO_MEHS_H

#include "simStructs.h"

#define JELLO_HEIGHT 5.0f
#define JELLO_RES 10
#define JELLO_SIZE 5

#define INDEX(i,j,k) i + j * (JELLO_RES + 1)  + k * (JELLO_RES + 1) * (JELLO_RES + 1)

class jello{
private:
	void initParticles();
	void addStructuralSprings(particle p1, particle p2);
	void addShearSprings(particle p1, particle p2);
	void addBendSprings(particle p1, particle p2);

public:
	jello();
	~jello();

	particle* particles;
	spring* structural;
	spring* shear;
	spring* bend;
};

#endif JELLO_MESH_H