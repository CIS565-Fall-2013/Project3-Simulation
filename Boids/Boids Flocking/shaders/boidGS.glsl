#version 330

uniform mat4 u_projMatrix;
uniform mat4 u_viewMatrix;
uniform vec3 u_lightPos;


layout (points) in;
layout (triangle_strip, max_vertices = 16) out;


in VertexData{
	vec3 EyeNormal;
	vec3 EyeForward;
	vec3 Color;
	float Length;
	float HalfWingSpan;
	float DeltaSweep;
	float HalfBodyHeight;
	float WingDeflection;
}vertexData[];


out vec3 fs_EyeNormal;
out vec3 fs_Color;
out vec3 fs_EyeLightVector;
out vec2 fs_TexCoord;



//From http://www.neilmendoza.com/glsl-rotation-about-an-arbitrary-axis/
mat4 rotationMatrix(vec3 axis, float angle)
{
    axis = normalize(axis);
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

	vec3 Up      = normalize(vertexData[0].EyeNormal.xyz);
	vec3 Forward = normalize(vertexData[0].EyeForward.xyz);
	vec3 Right   = normalize(cross(Forward, Up));

	mat4 Rotate = rotationMatrix(Forward, vertexData[0].WingDeflection);
	mat4 Rotate_T = transpose(Rotate);

	//Passthrough color
	fs_Color = vertexData[0].Color;
	
	vec3 EyeLightPos = vec3(u_viewMatrix*vec4(u_lightPos, 1.0));
	vec3 BackTopPos = Position + vertexData[0].HalfBodyHeight*Up;
	vec3 BackBottomPos = Position - vertexData[0].HalfBodyHeight*Up;
	vec3 RightWingtipPos = Position + vec3(Rotate   * vec4(-vertexData[0].DeltaSweep*Forward + vertexData[0].HalfWingSpan*Right,0.0));
	vec3 LeftWingtipPos  = Position + vec3(Rotate_T * vec4(-vertexData[0].DeltaSweep*Forward - vertexData[0].HalfWingSpan*Right,0.0));
	vec3 FrontPos = Position + vertexData[0].Length*Forward;
	
	vec3 RightTipUpNormal   = normalize(cross(FrontPos - RightWingtipPos,    BackTopPos-RightWingtipPos  ));
	vec3 RightTipDownNormal = normalize(cross(BackBottomPos-RightWingtipPos, FrontPos-RightWingtipPos    ));
	vec3 LeftTipUpNormal    = normalize(cross(BackTopPos-LeftWingtipPos,     FrontPos - LeftWingtipPos   ));
	vec3 LeftTipDownNormal  = normalize(cross(FrontPos-LeftWingtipPos,       BackBottomPos-LeftWingtipPos));
	vec3 TopNormal     = normalize((RightTipUpNormal  +LeftTipUpNormal  )/2.0);
	vec3 BottomNormal  = normalize((RightTipDownNormal+LeftTipDownNormal)/2.0);
	


	//=====Compute right wingtip======
	gl_Position = u_projMatrix*vec4(RightWingtipPos,1.0);
	fs_EyeNormal = RightTipUpNormal;
	fs_EyeLightVector =  EyeLightPos - RightWingtipPos;
	fs_TexCoord = vec2(1.0,0.0);
	EmitVertex();



	//=====Compute front point======
	gl_Position = u_projMatrix*vec4(FrontPos,1.0);
	fs_EyeNormal = TopNormal;
	fs_EyeLightVector =  EyeLightPos - FrontPos;
	fs_TexCoord = vec2(0.5,0.5);
	EmitVertex();



	//====Compute back center top point====
	gl_Position = u_projMatrix*vec4(BackTopPos,1.0);
	fs_EyeNormal = TopNormal;
	fs_EyeLightVector =  EyeLightPos - BackTopPos;
	fs_TexCoord = vec2(0.5,0.0);
	EmitVertex();



	//====Compute left wingtip====
	gl_Position = u_projMatrix*vec4(LeftWingtipPos,1.0);
	fs_EyeNormal = LeftTipUpNormal;
	fs_EyeLightVector =  EyeLightPos - LeftWingtipPos;
	fs_TexCoord = vec2(0.0,0.0);
	EmitVertex();


    EndPrimitive();
}