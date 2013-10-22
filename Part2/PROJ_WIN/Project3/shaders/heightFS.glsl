varying float f_height;
varying vec2 v_Texcoords;
attribute vec4 Position;

void main(void)
{
    float shade = 1.75*(1.0-2.0*sqrt(f_height));
	float alpha = float(mod(v_Texcoords.x+0.025, 0.05) > 0.023 || 
		                mod(v_Texcoords.y+0.025, 0.05) > 0.023);
    vec4 color = mix(vec4(0.05,0.3,0.4,1.0), vec4(0.05, 0.15, 0.3, 1.0), alpha);
	float dist = 5 * length(v_Texcoords);
	float light = (0.3 + 0.9 * exp(-dist));
	gl_FragColor = shade*color * light;
}