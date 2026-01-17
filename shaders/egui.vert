#version 450

layout(push_constant) uniform PushConstants {
    vec2 screen_size;
} push;

layout(location = 0) in vec2 in_pos;
layout(location = 1) in vec2 in_uv;
layout(location = 2) in vec4 in_color;

layout(location = 0) out vec2 frag_uv;
layout(location = 1) out vec4 frag_color;

void main() {
    // Convert from screen coordinates to NDC
    vec2 pos = 2.0 * in_pos / push.screen_size - 1.0;
    gl_Position = vec4(pos.x, pos.y, 0.0, 1.0);
    frag_uv = in_uv;
    // egui uses sRGB colors, pass through
    frag_color = in_color;
}
