#include "glm/glm.hpp"
#include "jelloMesh.h"
#include <math.h>

using namespace std;

jello::jello(){
	initParticles();
}

jello::~jello(){
}

void jello::initParticles(int size, int resolution){
	int numParticles = (JELLO_RES + 1) * (JELLO_RES + 1) * (JELLO_RES + 1);
	float distance = (float)JELLO_SIZE / (float)JELLO_RES;

	particles = new particle[numParticles];

	for(int i = 0; i < resolution + 1; i++){
		for(int j = 0; j < resolution + 1; j++){
			for(int k = 0; k < resolution + 1; k++){
				particle p;
				p.idx = INDEX(i,j,k);
				p.m = 1.0f;
				p.v = glm::vec3(0.0f);
				p.pos = glm::vec3((-(resolution + 1)/2 + k) * distance, (-(resolution + 1)/2 + j) * distance, distance * i);
				particles[INDEX(i,j,k)] = p;
			}
		}
	}
}

void jello::addStructuralSprings(particle p1, particle p2){
}

void jello::addShearSprings(particle p1, particle p2){
}

void jello::addBendSprings(particle p1, particle p2){
}

