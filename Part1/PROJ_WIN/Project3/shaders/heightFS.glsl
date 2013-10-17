varying float f_height;

void main(void)
{
    float shade = (1.0-2.0*sqrt(f_height));
    vec4 color = vec4(0.05,0.15,0.3,1.0); 
    gl_FragColor = shade*color;
}