#version 400

in vec2 g_Texcoords;
in float g_height;

noperspective in vec3 g_PatchDistance;
noperspective in vec3 g_TriDistance;

// noperspective in vec3 dist;

out vec4 FragColor;

float amplify(float d, float scale, float offset)
{
    d = scale * d + offset;
    d = clamp(d, 0, 1);
    d = 1 - exp2(-2*d*d);
    return d;
}

void main(void)
{

	//float nearD = min(min(dist.x,dist.y),dist.z);
	//float edgeIntensity = exp2(-1.0*nearD*nearD);
	
	//vec4 edgeColor = vec4(0.1,0.1,0.1,1.0);
	//color = edgeIntensity * edgeColor + (1-edgeIntensity) * color;


    float shade = (1.0-2.0*sqrt(g_height));

	float alpha = float(mod(g_Texcoords.x+0.025, 0.05) > 0.046 ||
						mod(g_Texcoords.y+0.025, 0.05) > 0.046);
    vec4 color;


    float d1 = min(min(g_TriDistance.x, g_TriDistance.y), g_TriDistance.z);
    float d2 = min(min(g_PatchDistance.x, g_PatchDistance.y), g_PatchDistance.z);
	alpha =  2.0f * (0.5f - amplify(d1, 40, -0.5) * amplify(d2, 60, -0.5));

    //color = amplify(d1, 40, -0.5) * amplify(d2, 60, -0.5) * color;
	color = mix(vec4(0.05,0.15,0.3,1.0), vec4(0.05, 0.3, 0.4, 1.0), alpha);

    FragColor = shade * color;
}