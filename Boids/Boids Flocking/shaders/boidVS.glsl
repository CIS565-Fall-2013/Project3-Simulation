#version 330

uniform mat4 u_projMatrix;
uniform vec3 u_lightPos;

in vec4 vs_position;
in vec4 vs_norm;
in vec4 vs_vel;
in vec3 vs_color;
in vec4 vs_shape; //Length, Wingspan, Delta, Deflection Angle

out VertexData{
	vec4 norm;
	vec4 vel;
	vec3 color;
	vec4 shape;//Length, Wingspan, Delta, Deflection Angle
}vertexData;

//Pass through to geometry shader for now. 
//Might do some transforms here later, but geom shader makes more sense
void main(void)
{
	gl_Position = vs_position;
	vertexData.norm = vs_norm;
	if(length(vs_vel) == 0)
		vertexData.vel = vec4(1,0,0,0);
	else
		vertexData.vel = vs_vel;
	vertexData.color = vs_color;
	vertexData.shape = vs_shape;
}
