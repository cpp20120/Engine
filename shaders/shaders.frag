#version 460

out vec4 outColour;
in  vec2 passTextureCoord;

uniform sampler2D texSampler;

void main()
{
    vec4 colour = texture(texSampler, passTextureCoord);

    outColour = colour;
}
