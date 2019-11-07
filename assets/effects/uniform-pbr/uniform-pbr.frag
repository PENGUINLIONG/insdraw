#version 450 core

layout(constant_id=0) const int NLIGHT = 1;

// # Input
//
// Interpolated fragment representation.
struct Repr {
    vec4 pos;
    vec4 normal;
    vec2 uv;
};
layout(location=0) in Repr repr;
//

// # Output
//
// Color value.
layout(location=0) out vec4 color;
//

// # Uniforms
//
// Light source information. Note that pos is a directional light when the w-
// component is 0 and a point light when the w-component is 1.
struct Light {
    vec3 pos;
    vec3 color;
};
layout(std140, set=1) readonly
uniform Lighting {
    vec4 cam_pos;
    Light[NLIGHT] lights;
};
//
// Material information.
layout(std140, binding=1)
uniform Material {
    vec3 albedo;
    float metalicity;
    float roughness;
};
//

const float PI = 3.1415926;
const float TAU = PI * 2;

void main() {
    color = vec4(0.0, 0.0, 1.0, 1.0);
}
