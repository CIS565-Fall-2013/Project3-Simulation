#version 330

uniform mat4 u_projMatrix;
uniform vec3 u_cameraPos;

layout (points) in;
layout (triangle_strip, max_vertices = 4) out;

in VertexData{
	vec4 norm;
	vec4 vel;
	vec3 color;
	vec4 shape;//Length, Wingspan, Delta, Deflection Angle
}vertexData[];

out vec3 Normal;
out vec3 Color;
out vec2 TexCoord;

void main()
{
    vec3 Position = gl_in[0].gl_Position.xyz;
    WorldCoord = Position;

    ToCam = normalize(u_cameraPos - Position);
    Up = vec3(0.0, 0.0, 1.0);
    Right = cross(ToCam, Up);
    Up = cross(Right, ToCam);

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

    Pos = Position - scale*Right + scale*Up;
    gl_Position = u_projMatrix * vec4(Pos, 1.0);
    TexCoord = vec2(1.0, 1.0);
    EmitVertex();

    EndPrimitive();
}