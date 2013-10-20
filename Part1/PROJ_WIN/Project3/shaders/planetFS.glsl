#version 330

in vec3 WorldCoord;
in vec3 ToCam;
in vec3 Up;
in vec3 Right;
in vec2 TexCoord;
flat in int gid;
out vec4 FragColor;

/*
http://stackoverflow.com/questions/4200224/random-noise-functions-for-glsl
*/

float rand(vec2 co){
    return fract(sin(dot(co.xy ,vec2(12.9898,78.233))) * 43758.5453);
}

void main()
{
	    vec2 coord = 2.01 * (TexCoord - vec2(0.5));
    float r = length(coord);
    if (r >= 1.0) { discard; }

    float dist = length(WorldCoord);
    if(dist <= 0.01)
    {
        FragColor = vec4(1.0);
        return;
    }

    vec3 N = Right*-coord.x + Up*coord.y + ToCam*sqrt(1-r*r);
    vec3 L = normalize(-WorldCoord);
    float light = 0.1 + 0.9*clamp(dot(N,L),0.0, 1.0)*exp(-dist);
    //vec3 color = vec3(0.4, 0.1, 0.6);
	float rCol1 = rand( vec2(gid*5346,gid*75676));
	float rCol2 = rand( vec2(gid*323132,gid*423423));
	float rCol3 = rand( vec2(gid*2312334,gid*32132));
	vec3 color = rCol1*vec3(1.0,0.0,0.0)+rCol2*vec3(0.0,1.0,0.0)+rCol3*vec3(0.0,0.0,1.0) + vec3(0.1,0.1,0.1);
    FragColor = vec4(color*light,1.0);
} 