#version 450 core

layout(location=0)
in vec2 pos;
layout(location=1)
in vec3 color;

layout(location=0)
out vec4 vColor;

void main() {
    vColor = vec4(color, 1.0);
    gl_Position = vec4(pos, 0.0, 1.0);
}
