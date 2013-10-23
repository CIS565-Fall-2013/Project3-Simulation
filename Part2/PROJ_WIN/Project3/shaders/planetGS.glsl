#version 330

uniform mat4 u_projMatrix;
uniform vec3 u_cameraPos;

layout (points) in;
layout (triangle_strip) out;
layout (max_vertices = 4) out;

out vec3 WorldCoord;
out vec3 ToCam;
out vec3 Up;
out vec3 Right;
out vec2 TexCoord;

const float scale = 0.03;

void main()
{
    vec3 Position = gl_in[0].gl_Position.xyz;
    WorldCoord = Position;

    ToCam = normalize(u_cameraPos - Position);
  /*  Up = vec3(0.0, 0.0, 1.0);
    Right = cross(ToCam, Up);
    Up = cross(Right, ToCam);*/
	float theta = gl_in[0].gl_Position.w;
	vec3 column0 = vec3(cos(theta), sin(theta), 0);
	vec3 column1 = vec3(-sin(theta), cos(theta), 0);
	vec3 column2 = vec3(0, 0, 1);
	mat3 rotZ = mat3(column0, column1, column2);
	Right = rotZ * vec3(1.0, 0.0, 0.0);
	Up = rotZ * vec3(0.0,1.0,0.0);

	vec3 Pos = Position + scale*Right - scale*Up;
    gl_Position = u_projMatrix * vec4(Pos, 1.0);
    TexCoord = vec2(0.0, 0.0);
    EmitVertex();

    Pos = Position + scale*Right + scale*Up;
    gl_Position = u_projMatrix * vec4(Pos, 1.0);
    TexCoord = vec2(0.0, 1.0);
    EmitVertex();

    Pos = Position - scale*Right - scale*Up;
    gl_Position = u_projMatrix * vec4(Pos, 1.0);
    TexCoord = vec2(1.0, 0.0);
    EmitVertex();

    //Pos = Position - scale*Right + scale*Up;
    //gl_Position = u_projMatrix * vec4(Pos, 1.0);
    //TexCoord = vec2(1.0, 1.0);
    //EmitVertex();

    EndPrimitive();
}
