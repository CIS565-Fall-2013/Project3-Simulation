#version 330

uniform mat4 u_projMatrix;
uniform vec3 u_cameraPos;

in vec4 vel[1];
in vec4 pos[1];

layout (points) in;
layout (triangle_strip) out;
layout (max_vertices = 3) out;

out vec3 WorldCoord;
out vec3 ToCam;
out vec3 Up;
out vec3 Right;
out vec2 TexCoord;

const float scale = 0.03;

void main()
{
    vec3 Position = vec3(pos[0].x, pos[0].y, pos[0].z);
	WorldCoord = Position;
	//vec3 Velocity = vec3(1,0,0);
	vec3 Velocity = vec3(vel[0].x, vel[0].y, vel[0].z);
	vec3 Direction = normalize(Velocity);

    ToCam = normalize(u_cameraPos - Position);
    Up = vec3(0.0, 0.0, 1.0);
    Right = cross(Direction, Up);
	if(length(Right) == 0){
		Right = cross(Direction, ToCam);
	}

	vec3 Pos = Position + scale * 0.5 * Right;
    gl_Position = u_projMatrix * vec4(Pos, 1.0);
    TexCoord = vec2(0.0, 0.0);
    EmitVertex();

    Pos = Position - scale * 0.5 * Right;
    gl_Position = u_projMatrix * vec4(Pos, 1.0);
    TexCoord = vec2(0.0, 1.0);
    EmitVertex();

    Pos = Position - scale * Direction;
    gl_Position = u_projMatrix * vec4(Pos, 1.0);
    TexCoord = vec2(1.0, 0.0);
    EmitVertex();

    EndPrimitive();
}