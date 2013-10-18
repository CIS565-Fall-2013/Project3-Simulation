/*
void main(void)
{
	gl_FragColor = vec4(0.05,0.15,0.3,1.0);
}
*/

varying vec2 v_Texcoords;
varying float f_height;

void main(void)
{
    float shade = (1.0-2.0*sqrt(f_height));
    float alpha = float(mod(v_Texcoords.x+0.025, 0.05) > 0.046 || mod(v_Texcoords.y+0.025, 0.05) > 0.046);
    vec4 color = mix(vec4(0.05,0.15,0.3,1.0), vec4(0.05, 0.3, 0.4, 1.0), alpha);

    //vec4 color = vec4(0.05,0.15,0.3,1.0); 
    gl_FragColor = shade*color;
}
