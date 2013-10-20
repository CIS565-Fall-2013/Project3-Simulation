#version 330

uniform mat4 u_projMatrix;
uniform vec3 u_cameraPos;

layout (points) in;
layout (triangle_strip) out;
layout (max_vertices = 6) out;

in vec2 gs_Velocity[];

out vec3 WorldCoord;
out vec3 WorldVec;
out vec3 Up;
out vec3 Right;
out vec3 TexCoord;

out vec4 tag;

const float scale = 0.02;

void main()
{
    vec3 Position = gl_in[0].gl_Position.xyz;
    WorldCoord = Position;
	
	// Initialize
    WorldVec = vec3(normalize(gs_Velocity[0]), 0.0);
    Up = vec3(0.0, 0.0, 1.0);
    Right = cross(WorldVec, Up);     // y
    Up = cross(Right, WorldVec);     // x

	// A
	vec3 Pos = Position + scale / 2 * Up - scale * sqrt(3)/ 2 * Right ;
    gl_Position = u_projMatrix * vec4(Pos, 1.0);
    TexCoord = normalize(cross(vec3(1/2, sqrt(3)/ 2, 0) - vec3(-3/2, 0, 0), vec3(-3/2, 0, 0) - vec3(0, 0, 3)));
	tag = vec4(1.0, 0.0, 0.0, 0.0);
    EmitVertex();

	// B
	Pos = Position  + scale / 2 * Up + scale * sqrt(3)/ 2 * Right ;
    gl_Position = u_projMatrix * vec4(Pos, 1.0);
    TexCoord = normalize(cross(vec3(1/2, -sqrt(3)/ 2, 0) - vec3(-3/2, 0, 0), vec3(-3/2, 0, 0) - vec3(0, 0, 3)));
	tag = vec4(0.0, 1.0, 0.0, 0.0);
    EmitVertex();

	// C
    Pos = Position - 3 / 2 * scale * Up;
    gl_Position = u_projMatrix * vec4(Pos, 1.0);
    TexCoord = normalize(cross(vec3(0, sqrt(3), 0), vec3(1/2, sqrt(3)/ 2, 0) - vec3(0, 0, 3)));
	tag = vec4(0.0, 0.0, 1.0, 0.0);
    EmitVertex();

	// D
    Pos = Position + 3*scale * WorldVec;
    gl_Position = u_projMatrix * vec4(Pos, 1.0);
	TexCoord = vec3(0, 0, 1);
	tag = vec4(0.0, 0.0, 0.0, 1.0);
    EmitVertex();

	// A
    Pos = Position + scale / 2 * Up - scale * sqrt(3)/ 2 * Right ;
    gl_Position = u_projMatrix * vec4(Pos, 1.0);
    TexCoord = normalize(cross(vec3(1/2, sqrt(3)/ 2, 0) - vec3(-3/2, 0, 0), vec3(-3/2, 0, 0) - vec3(0, 0, 3)));
	tag = vec4(1.0, 0.0, 0.0, 0.0);
    EmitVertex();

	// B
    Pos = Position  + scale / 2 * Up + scale * sqrt(3)/ 2 * Right ;
    gl_Position = u_projMatrix * vec4(Pos, 1.0);
    TexCoord = normalize(cross(vec3(1/2, -sqrt(3)/ 2, 0) - vec3(-3/2, 0, 0), vec3(-3/2, 0, 0) - vec3(0, 0, 3)));
	tag = vec4(0.0, 1.0, 0.0, 0.0);
    EmitVertex();

    EndPrimitive();
}