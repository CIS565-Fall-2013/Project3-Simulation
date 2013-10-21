#version 330

uniform vec3 u_targetposition;

in vec3 WorldCoord;
in vec3 ToCam;
in vec3 Up;
in vec3 Right;
in vec2 TexCoord;
out vec4 FragColor;


in vec3 normal_f1;
in vec3 normal_f2;
in vec3 normal_f3;
in vec3 normal_f4;

in vec3 thecolor;

void main()
{
	    vec2 coord = 2.01 * (TexCoord - vec2(0.5));
    float r = length(coord);
 //   if (r >= 1.0) { discard; }
	float dist = length(-u_targetposition/100.0f-WorldCoord);
    if(dist <= 0.01)
    {
        FragColor = vec4(1.0f);
        return;
    }

	vec3 N = Right*-coord.x + Up*coord.y + ToCam*sqrt(1-r*r);

	float x=TexCoord.x;
	float y=TexCoord.y;
	if(y<0.5f){if(x+y<0.5f) N=normal_f1;else N=normal_f2;}
	else{if(x+y<1.0f)N=normal_f3;else N=normal_f4;}

	vec3 L2= vec3(0,0,1);
    float light2 = 0.1 + 0.9*clamp(dot(N,L2),0.0, 1.0);//*exp(-dist);


    vec3 L1 = normalize(-u_targetposition/100.0f-WorldCoord);
    float light1 = 0.1 + 0.9*clamp(dot(N,L1),0.0, 1.0)*exp(-dist/2.0f);



    vec3 color = vec3(77, 184, 73)*(1.0f/255.0f);
	
    FragColor = vec4(thecolor*(light1+light2),1.0)*1.0f;
	//FragColor=vec4((N+vec3(1.0f))*0.5f,1.0f);

	
} 