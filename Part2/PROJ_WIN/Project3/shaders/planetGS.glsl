#version 330

// want to create a geometry shader that takes in points and emits triangle strips
// sample geometry shader: http://www.lighthouse3d.com/tutorials/glsl-core-tutorial/geometry-shader/

uniform mat4 u_projMatrix;
uniform vec3 u_cameraPos;

layout (points) in;
layout (triangle_strip) out;
layout (max_vertices = 3) out;

in vColor
{
	vec3 color;
	vec3 velocity;
}vertices[];

out fColor
{
	vec3 color;
}frag;

out vec3 WorldCoord;
out vec3 ToCam;
out vec3 Up;
out vec3 Right;
out vec2 TexCoord;

const float scale = 0.02;

void main()
{
	int i;
	vec3 orientation;

	for (int i = 0 ; i < gl_in.length() ; ++i) // gl_in.length() is 1
	{
		orientation = vertices[i].velocity;
		frag.color = vertices[i].color;
	}

    vec3 Position = gl_in[0].gl_Position.xyz;
    WorldCoord = Position;

    ToCam = normalize(u_cameraPos - Position);
    Up = vec3(0.0, 0.0, 1.0); // note that z axis is the up vector here
    Right = cross(orientation, Up);
    Up = cross(Right, orientation);

	//vec3 Pos = Position + scale*Right - scale*Up;
	//gl_Position = u_projMatrix * vec4(Pos, 1.0);
	//TexCoord = vec2(0.0, 0.0);
	//EmitVertex();

    //Pos = Position + scale*Right + scale*Up;
    //gl_Position = u_projMatrix * vec4(Pos, 1.0);
    //TexCoord = vec2(0.0, 1.0);
    //EmitVertex();

    //Pos = Position - scale*Right - scale*Up;
    //gl_Position = u_projMatrix * vec4(Pos, 1.0);
    //TexCoord = vec2(1.0, 0.0);
    //EmitVertex();

    //Pos = Position - scale*Right + scale*Up;
    //gl_Position = u_projMatrix * vec4(Pos, 1.0);
    //TexCoord = vec2(1.0, 1.0);
    //EmitVertex();

    vec3 Pos = Position + scale*Right - scale*Up;
	gl_Position = u_projMatrix * vec4(Pos, 1.0);
	TexCoord = vec2(0.0, 0.0);
	EmitVertex();
	
	Pos = Position - scale*Right - scale*Up;
    gl_Position = u_projMatrix * vec4(Pos, 1.0);
    TexCoord = vec2(1.0, 0.0);
    EmitVertex();

    Pos = Position - scale*Right + scale*Up;
    gl_Position = u_projMatrix * vec4(Pos, 1.0);
    TexCoord = vec2(0.5, 1);
    EmitVertex();

	
	EndPrimitive();
}