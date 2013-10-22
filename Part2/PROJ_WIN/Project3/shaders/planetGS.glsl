#version 330

uniform mat4 u_projMatrix;
uniform vec3 u_cameraPos;

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
    vec3 Position = vec3(gl_in[0].gl_Position.xy, 0.0);
	float angle = gl_in[0].gl_Position.z;

    WorldCoord = Position;

    ToCam = normalize(u_cameraPos - Position);
    Up = vec3(0.0, 0.0, 1.0);
    Right = cross(ToCam, Up);
    Up = cross(Right, ToCam);

	float p0_x = 2.0;
	float p0_y = 0.0;
	float p1_x = 0.0;
	float p1_y = 0.5;
	float p2_x = 0.0;
	float p2_y = -0.5;

	float p0_xx = p0_x * cos(angle) - p0_y * sin(angle);
	float p0_yy = p0_x * sin(angle) + p0_y * cos(angle);
	float p1_xx = p1_x * cos(angle) - p1_y * sin(angle);
	float p1_yy = p1_x * sin(angle) + p1_y * cos(angle);
	float p2_xx = p2_x * cos(angle) - p2_y * sin(angle);
	float p2_yy = p2_x * sin(angle) + p2_y * cos(angle);

	vec3 Pos = Position + scale*vec3(p0_xx, p0_yy, 1.0);
    gl_Position = u_projMatrix * vec4(Pos, 1.0);
    TexCoord = vec2(0.0, 0.0);
    EmitVertex();

    Pos = Position + scale*vec3(p1_xx, p1_yy, 1.0);
    gl_Position = u_projMatrix * vec4(Pos, 1.0);
    TexCoord = vec2(0.0, 1.0);
    EmitVertex();

    Pos = Position + scale*vec3(p2_xx, p2_yy, 1.0);
    gl_Position = u_projMatrix * vec4(Pos, 1.0);
    TexCoord = vec2(1.0, 0.0);
    EmitVertex();

    EndPrimitive();
}