#version 330

uniform mat4 u_projMatrix;
uniform vec3 u_cameraPos;

layout (points) in;
layout (triangle_strip) out;
layout (max_vertices = 4) out;

in vec4 velocity[1];

out vec3 WorldCoord;
out vec3 ToCam;
out vec3 Up;
out vec3 Right;
out vec2 TexCoord;

out vec3 fs_color;

const float scale = 0.03;

void main()
{
    vec3 Position = gl_in[0].gl_Position.xyz;
    WorldCoord = Position;

	vec3 direction = normalize(velocity[0].xyz);
	
	fs_color = normalize(Position);

	ToCam = normalize(u_cameraPos - Position);
	Up = vec3(0.0f, 0.0f, 1.0f);
	Right = cross(Up, direction);

	if(dot(direction, Up) < 0.001f){
		Right = cross(direction, ToCam);
	}

	//find points of triangle
	vec3 Pos = Position - 2.0f*scale*direction;
	gl_Position = u_projMatrix * vec4(Pos, 1.0);
	//TexCoord = vec2(0.0, 0.0);
	EmitVertex();

    Pos = Position - 0.5*scale*Right;
    gl_Position = u_projMatrix * vec4(Pos, 1.0);
    //TexCoord = vec2(0.0, 1.0);
    EmitVertex();

    Pos = Position + 0.5*scale*Right;
    gl_Position = u_projMatrix * vec4(Pos, 1.0);
    //TexCoord = vec2(1.0, 0.0);
    EmitVertex();

    EndPrimitive();
}