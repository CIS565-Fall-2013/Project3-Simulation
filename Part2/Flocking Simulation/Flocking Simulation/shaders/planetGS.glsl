#version 330

uniform mat4 u_projMatrix;
uniform vec3 u_cameraPos;

layout (points) in;
layout (triangle_strip, max_vertices = 12) out;

in vec3[] gVelocity;

out vec3 worldCoord;
out vec3 worldNormal;
out vec3 fVelocity;

const float scale = 0.03;
void main()
{
    vec3 Position = gl_in[0].gl_Position.xyz;
    worldCoord = Position;

	vec3 velocity = normalize(gVelocity[0]);
	fVelocity = velocity;

	vec3 right = normalize(cross(velocity, vec3(0.0, 1.0, 0.0)));
	vec3 up = cross(right, velocity);

	vec3 Pos1 = Position + 0.1*velocity;
	vec3 Pos2 = Position + 0.1*(0.3*-right - 0.3*up - 0.2*velocity);
	vec3 Pos3 = Position + 0.1*(0.3*up - 0.3*velocity);
	vec3 Pos4 = Position + 0.1*(0.3*right - 0.3*up - 0.2*velocity);

	gl_Position = u_projMatrix * vec4(Pos1, 1.0);
	worldNormal = normalize(cross(Pos2- Pos1, Pos3 - Pos1));
	EmitVertex();

	gl_Position = u_projMatrix * vec4(Pos2, 1.0);
	worldNormal = normalize(cross(Pos2- Pos1, Pos3 - Pos1));
	EmitVertex();

	gl_Position = u_projMatrix * vec4(Pos3, 1.0);
	worldNormal = normalize(cross(Pos2- Pos1, Pos3 - Pos1));
	EmitVertex();

	EndPrimitive();

	gl_Position = u_projMatrix * vec4(Pos2, 1.0);
	worldNormal = normalize(cross(Pos4- Pos2, Pos3 - Pos2));
	EmitVertex();

	gl_Position = u_projMatrix * vec4(Pos4, 1.0);
	worldNormal = normalize(cross(Pos4- Pos2, Pos3 - Pos2));
	EmitVertex();

	gl_Position = u_projMatrix * vec4(Pos3, 1.0);
	worldNormal = normalize(cross(Pos4- Pos2, Pos3 - Pos2));
	EmitVertex();

	EndPrimitive();

	gl_Position = u_projMatrix * vec4(Pos3, 1.0);
	worldNormal = normalize(cross(Pos4- Pos3, Pos1 - Pos3));
	EmitVertex();

	gl_Position = u_projMatrix * vec4(Pos4, 1.0);
	worldNormal = normalize(cross(Pos4- Pos3, Pos1 - Pos3));
	EmitVertex();

	gl_Position = u_projMatrix * vec4(Pos1, 1.0);
	worldNormal = normalize(cross(Pos4- Pos3, Pos1 - Pos3));
	EmitVertex();

	EndPrimitive();

	gl_Position = u_projMatrix * vec4(Pos1, 1.0);
	worldNormal = normalize(cross(Pos4- Pos1, Pos2 - Pos1));
	EmitVertex();

	gl_Position = u_projMatrix * vec4(Pos4, 1.0);
	worldNormal = normalize(cross(Pos4- Pos1, Pos2 - Pos1));
	EmitVertex();

	gl_Position = u_projMatrix * vec4(Pos2, 1.0);
	worldNormal = normalize(cross(Pos4- Pos1, Pos2 - Pos1));
	EmitVertex();

	EndPrimitive();
}
