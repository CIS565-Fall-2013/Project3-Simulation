#ifndef CONSTRAINTS_H
#define CONSTRAINTS_H

struct Edge
{
    int v1, v2;
    int tri1, tri2;
	__host__ __device__ Edge(int ev1, int ev2, int et1, int et2) :
		v1(ev1), v2(ev2), tri1(et1), tri2(et2) {}
};

struct FixedPointConstraint
{
	int v0;
	glm::vec3 fixed_point;
	__host__ __device__ FixedPointConstraint(int v, glm::vec3 p) :
		v0(v), fixed_point(p) {}
};

struct StretchConstraint
{
	int v1, v2;
	float rest_length;
	__host__ __device__ StretchConstraint(int v01, int v02, float l) :
		v1(v01), v2(v02), rest_length(l) {}
};

struct BendConstraint
{
	int v1, v2, v3, v4;
	float phi;
	__host__ __device__ BendConstraint(int v01, int v02, int v03, int v04, float p) :
		v1(v01), v2(v02), v3(v03), v4(v04), phi(p) {}
};

struct CollisionConstraint
{
	unsigned int v0;
	glm::vec3 ref_point, normal;
	__host__ __device__ CollisionConstraint(int v, glm::vec3 p, glm::vec3 n) :
		v0(v), ref_point(p), normal(n) {}
};

struct SelfCollisionConstraint
{
	float stiff;
	unsigned int u, v1, v2, v3;
	float h;
	__host__ __device__ SelfCollisionConstraint(int u0, int v01, int v02, int v03, float h0) :
		u(u0), v1(v01), v2(v02), v3(v03), h(h0) {}
};

#endif