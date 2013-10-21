varying float f_height;

varying vec2 v_Texcoords;

void main(void)
{
    float shade = (1.0-2.0*sqrt(f_height));
    float alpha = float(mod(v_Texcoords.x+0.025, 0.05) > 0.046 ||
                    mod(v_Texcoords.y+0.025, 0.05) > 0.046);
    vec4 color = mix(vec4(0.05,0.15,0.3,1.0), vec4(0.05, 0.3, 0.4, 1.0), alpha);
    gl_FragColor = shade*color;
}