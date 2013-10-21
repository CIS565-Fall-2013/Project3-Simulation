#version 330

uniform mat4 u_projMatrix;
uniform mat4 u_viewMatrix;
uniform vec3 u_lightPos;


layout (points) in;
layout (triangle_strip, max_vertices = 4) out;


in VertexData{
	vec3 EyeNormal;
	vec3 EyeForward;
	vec3 Color;
	float Length;
	float HalfWingSpan;
	float DeltaSweep;
	float WingDeflection;
}vertexData[];


varrying vec3 fs_EyeNormal;
varrying vec3 fs_Color;
varrying vec3 fs_EyeLightVector;
varrying vec2 fs_TexCoord;



//From http://www.neilmendoza.com/glsl-rotation-about-an-arbitrary-axis/
mat4 rotationMatrix(vec3 axis, float angle)
{
    vec3 axis = normalize(axis);
    float s = sin(angle);
    float c = cos(angle);
    float oc = 1.0 - c;
    
    return mat4(oc * axis.x * axis.x + c,           oc * axis.x * axis.y - axis.z * s,  oc * axis.z * axis.x + axis.y * s,  0.0,
                oc * axis.x * axis.y + axis.z * s,  oc * axis.y * axis.y + c,           oc * axis.y * axis.z - axis.x * s,  0.0,
                oc * axis.z * axis.x - axis.y * s,  oc * axis.y * axis.z + axis.x * s,  oc * axis.z * axis.z + c,           0.0,
                0.0,                                0.0,                                0.0,                                1.0);
}


void main()
{
    vec3 Position = gl_in[0].gl_Position.xyz;
    VertexData vData = vertexData[0];

	vec3 Up      = normalize(vData.EyeNormal.xyz);
	vec3 Forward = normalize(vData.EyeForward.xyz);
	vec3 Right   = cross(Forward, Up);

	//Passthrough color
	fs_Color = vData.color;
	
	vec3 EyeLightPos = u_viewMatrix*u_lightPos;


	//=====Compute right wingtip======
	vec3 Pos = Position - delta*Forward + wingspan/2.0*Right;
	mat4 Rotate = rotationMatrix(Forward, -vData.WingDeflection);

	fs_EyeNormal = vec3(Rotate*vec4(Up,0.0));
	Pos = vec3(Rotate*vec4(Pos,0.0));
	gl_Position = u_projMatrix*vec4(Pos,1.0);
	fs_EyeLightVector =  EyeLightPos - Pos;

	TexCoord = vec2(1.0,0.0);
	EmitVertex();

	//=====Compute front point======
	Pos = Position + length*Forward;
	gl_Position = u_projMatrix*vec4(Pos,1.0);
	fs_EyeNormal = Up;
	fs_EyeLightVector =  EyeLightPos - Pos;
	TexCoord = vec2(0.5,1.0);
    EmitVertex();

	//====Compute back center point====
	Pos = Position;
	gl_Position = u_projMatrix*vec4(Pos,1.0);
	//fs_EyeNormal = Up; //Same as above
	fs_EyeLightVector =  EyeLightPos - Pos;
	TexCoord = vec2(0.5,0.0);
	EmitVertex();


	//====Compute left wingtip====
	Pos = Position - delta*Forward - wingspan/2.0*Right;
	Rotate = transpose(Rotate);//Inverse rotation for other wing

	EyeNormal = vec3(Rotate*vec4(Up,0.0));
	Pos = vec3(Rotate*vec4(Pos,0.0));

	gl_Position = u_projMatrix*vec4(Pos,1.0);
	fs_EyeLightVector =  EyeLightPos - Pos;
	TexCoord = vec2(0.0,0.0);
	EmitVertex();


    EndPrimitive();
}