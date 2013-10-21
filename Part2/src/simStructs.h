#ifndef SIM_STRUCTS_H
#define SIM_STRUCTS_H

#include "glm/glm.hpp"

struct boid{
	glm::vec3 pos;
	glm::vec3 vel;
	int groupIdx;
	float r;
};

#endif