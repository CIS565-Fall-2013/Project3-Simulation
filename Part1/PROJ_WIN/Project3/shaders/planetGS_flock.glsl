#version 330

uniform mat4 u_projMatrix;
uniform vec3 u_cameraPos;

layout (points) in;
layout (triangle_strip , max_vertices=6) out;

in vec3 v_Velocity[];

out vec3 WorldCoord;
out vec3 ToCam;
out vec3 Up;
out vec3 Right;
out vec2 TexCoord;

out vec3 normal_f1;
out vec3 normal_f2;
out vec3 normal_f3;
out vec3 normal_f4;

out vec3 thecolor;

const float scale = 0.03;


void main()
{

	vec3 g_Vel=v_Velocity[0];

    vec3 Position = gl_in[0].gl_Position.xyz;

    WorldCoord = Position;

	float coloridx=gl_in[0].gl_Position.w;
	if(coloridx<0.05f)
	{
		thecolor=vec3(227,51,49)/255.0f;
	}
	else if(coloridx<1.05f)
	{
		thecolor=vec3(252,210,9)/255.0f;
	}
	else if(coloridx<2.05f)
	{
		thecolor=vec3(76,183,73)/255.0f;
	}
	else if(coloridx<3.05f)
	{
		thecolor=vec3(78,135,192)/255.0f;
	}
	else if(coloridx<4.05f)
	{
		thecolor=vec3(163,101,247)/255.0f;
	}

	//thecolor=vec3(1,1,1);


	vec3 orientation=normalize(g_Vel);
	vec3 up=(abs(orientation.x)>0.5f)?vec3(0,1,0):vec3(1,0,0);
	vec3 left=normalize(cross(orientation,up));
	up=normalize(cross(orientation,left));


	vec3 pos1=Position+scale*(up);
	vec3 pos2=Position+scale*(-0.5*up-0.866*left);
	vec3 pos3=Position+scale*(-0.5*up+0.866*left);
	vec3 pos4=Position+scale*(4.414*orientation);




	normal_f1=normalize(cross(pos1-pos2,pos2-pos3));
	normal_f2=normalize(cross(pos4-pos2,pos2-pos3));
	normal_f3=normalize(cross(pos3-pos1,pos1-pos4));
	normal_f4=normalize(cross(pos4-pos1,pos1-pos2));


    ToCam = normalize(u_cameraPos - Position);
    Up = vec3(0.0, 0.0, 1.0);
    Right = cross(ToCam, Up);
    Up = cross(Right, ToCam);


	vec3 Pos = Position + scale*Right - scale*Up;
	Pos=pos1;
    gl_Position = u_projMatrix * vec4(Pos, 1.0);
    TexCoord = vec2(0.0, 0.0);
    EmitVertex();

    //Pos = Position + scale*Right + scale*Up;
	Pos=pos2;
    gl_Position = u_projMatrix * vec4(Pos, 1.0);
    TexCoord = vec2(0.5, 0.0);
    EmitVertex();

    //Pos = Position - scale*Right - scale*Up;
	Pos=pos3;
    gl_Position = u_projMatrix * vec4(Pos, 1.0);
    TexCoord = vec2(0.0, 0.5);
    EmitVertex();

	Pos=pos4;
    gl_Position = u_projMatrix * vec4(Pos, 1.0);
    TexCoord = vec2(0.5, 0.5);
    EmitVertex();

	Pos=pos1;
    gl_Position = u_projMatrix * vec4(Pos, 1.0);
    TexCoord = vec2(0.0, 1.0);
    EmitVertex();

	Pos=pos2;
    gl_Position = u_projMatrix * vec4(Pos, 1.0);
    TexCoord = vec2(0.5, 1.0);
    EmitVertex();

    //Pos = Position - scale*Right + scale*Up;
    //gl_Position = u_projMatrix * vec4(Pos, 1.0);
    //TexCoord = vec2(1.0, 1.0);
    //EmitVertex();

    EndPrimitive();
}
