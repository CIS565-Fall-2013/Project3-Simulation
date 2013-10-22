#version 400

layout (triangles) in;
layout (triangle_strip) out;
layout (max_vertices = 3) out;

in vec2 te_Texcoords[3];
in float te_height[3];
in vec3 te_PatchDistance[3];

out vec2 g_Texcoords;
out float g_height;
noperspective out vec3 g_PatchDistance;
noperspective out vec3 g_TriDistance;

uniform vec2 windowDim;
//noperspective out vec3 dist;

// Referencing 
// http://www2.imm.dtu.dk/~janba/Wireframe/
// http://prideout.net/blog/?p=48

void main()
{
/*
	vec2 p0 = windowDim * gl_in[0].gl_Position.xy / gl_in[0].gl_Position.w;
	vec2 p1 = windowDim * gl_in[1].gl_Position.xy / gl_in[1].gl_Position.w;
	vec2 p2 = windowDim * gl_in[2].gl_Position.xy / gl_in[2].gl_Position.w;

	vec2 v0 = p2-p1;
	vec2 v1 = p2-p0;
	vec2 v2 = p1-p0;
	float area = abs(v1.x*v2.y - v1.y * v2.x);
*/
//	dist = vec3(area/length(v0),0,0);
	g_Texcoords = te_Texcoords[0];
	g_height = te_height[0];
	g_PatchDistance = te_PatchDistance[0];
	g_TriDistance = vec3(1,0,0);
	gl_Position = gl_in[0].gl_Position;
	EmitVertex();

//	dist = vec3(0,area/length(v1),0);
	g_Texcoords = te_Texcoords[1];
	g_height = te_height[1];
	g_PatchDistance = te_PatchDistance[1];
	g_TriDistance = vec3(0,1,0);
	gl_Position = gl_in[1].gl_Position;
	EmitVertex();

//	dist = vec3(0,0,area/length(v2));
	g_Texcoords = te_Texcoords[2];
	g_height = te_height[2];
	g_PatchDistance = te_PatchDistance[1];
	g_TriDistance = vec3(0,0,1);
	gl_Position = gl_in[2].gl_Position;
	EmitVertex();

	EndPrimitive();
}