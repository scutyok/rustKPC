#version 450

// Input from vertex shader
layout(location = 0) in vec3 fragColor;
layout(location = 1) in vec2 fragTexCoord;
layout(location = 2) in vec3 fragWorldPos;
layout(location = 3) in vec3 fragNormal;

// Uniforms
layout(binding = 1) uniform sampler2D texSampler;

// Point light structure (matches Rust GpuLight)
struct PointLight {
    vec4 positionRadiusSq;  // xyz = position, w = radius²
    vec4 colorIntensity;    // rgb = color * intensity (pre-multiplied), w = invRadius
};

// Shadow caster structure (matches Rust GpuShadowCaster)
struct ShadowCaster {
    vec4 positionRadius;    // xyz = position (world space), w = radius
};

// Lighting UBO
layout(binding = 2) uniform LightingData {
    vec4 cameraPos;       // xyz = camera world position
    vec4 ambient;         // rgb = ambient light color
    vec4 fogColor;        // rgb = fog colour
    vec4 fogParams;       // x = near, y = far, z = 1.0 enabled / 0.0 disabled, w = skyFogFar
    uint lightCount;
    uint _pad0;
    uint _pad1;
    uint _pad2;
    PointLight lights[128];
    uint shadowCount;
    uint _pad3;
    uint _pad4;
    uint _pad5;
    ShadowCaster shadowCasters[32];
} lighting;

// Push constants for opacity
layout(push_constant) uniform PushConstants {
    layout(offset = 64) float opacity;
    // Sky-only UV offsets.  Kept as scalar floats so their byte layout is
    // explicit after the opacity float.
    layout(offset = 68) float skyUOffset;
    layout(offset = 72) float skyVOffset;
} push;

// Output
layout(location = 0) out vec4 outColor;

void main() {
    vec2 texCoord = fragTexCoord;
    if (push.opacity < 0.0) {
        texCoord += vec2(push.skyUOffset, push.skyVOffset);
    }
    vec4 texColor = texture(texSampler, texCoord);

    // Alpha test: discard fully transparent fragments (fences, grates, etc.)
    if (texColor.a < 0.5) {
        discard;
    }

    // Sky mode: negative opacity = sky (no lighting), magnitude = alpha
    if (push.opacity < 0.0) {
        vec4 skyColor = vec4(texColor.rgb, texColor.a * abs(push.opacity));
        // Only the opaque skybox receives fog.  Its geometry is scaled far
        // beyond the playable map, so use a much longer range than world fog
        // and calculate the fade from the player's actual distance.
        if (abs(push.opacity) > 0.99 && lighting.fogParams.z > 0.5) {
            float baseFar = max(lighting.fogParams.y, lighting.fogParams.w);
            // Start beyond normal map fog, but well before the outer sky
            // faces, so bounded skyboxes visibly receive the distance fade.
            float skyFogNear = baseFar * 2.0;
            float skyFogFar = baseFar * 25.0;
            float skyDistance = length(fragWorldPos - lighting.cameraPos.xyz);
            float skyFogAmount = smoothstep(skyFogNear, skyFogFar, skyDistance);
            skyColor.rgb = mix(skyColor.rgb, lighting.fogColor.rgb, skyFogAmount);
        }
        outColor = skyColor;
        return;
    }

    // Normal is already normalized from vertex shader
    vec3 N = fragNormal;

    // Use pre-baked vertex light as the base illumination.
    // fragColor = surface.colour/255 for BSP geometry (pre-compiled lighting from the .dat file).
    // fragColor = (1,1,1) for ABC placed objects (no pre-baked data — they stay fully lit).
    vec3 totalLight = fragColor;

    // lightCount is uniform across the draw call — clamp once so the loop bound
    // is a compile-time-visible constant range for the driver optimizer.
    int count = int(min(lighting.lightCount, 128u));
    for (int i = 0; i < count; i++) {
        vec3  lightPos   = lighting.lights[i].positionRadiusSq.xyz;
        float radiusSq   = lighting.lights[i].positionRadiusSq.w;
        vec3  lightColI  = lighting.lights[i].colorIntensity.rgb; // pre-multiplied color*intensity
        float invRadius  = lighting.lights[i].colorIntensity.w;

        vec3  toLight = lightPos - fragWorldPos;
        float distSq  = dot(toLight, toLight);

        // Early out using squared distance (no sqrt)
        if (distSq >= radiusSq) continue;

        // inversesqrt is a single HW instruction on most GPUs
        float invDist = inversesqrt(distSq);

        vec3  L = toLight * invDist;

        // Smooth quadratic attenuation: t = 1 - dist/radius = 1 - dist*invRadius
        float dist = distSq * invDist;  // dist = distSq / sqrt(distSq)
        float t = 1.0 - dist * invRadius;
        float atten = t * t;

        // Diffuse: two-sided so geometry with flipped normals still receives light
        float NdotL = abs(dot(N, L));

        totalLight += lightColI * NdotL * atten;
    }

    // ── Blob shadows (Lithtech-style projected downward) ────────────────
    // Like the original engine, shadows project straight down (dir = -Y).
    // Multiplicative darkening on upward-facing surfaces beneath shadow casters.
    float shadowFactor = 1.0;
    int nShadows = int(min(lighting.shadowCount, 32u));
    for (int s = 0; s < nShadows; s++) {
        vec3  casterPos    = lighting.shadowCasters[s].positionRadius.xyz;
        float shadowRadius = lighting.shadowCasters[s].positionRadius.w;

        // Only shadow fragments below the caster (Y is up in Vulkan coords)
        float heightDiff = casterPos.y - fragWorldPos.y;
        if (heightDiff < 0.0 || heightDiff > 3.0) continue;

        // Horizontal distance from caster center
        vec2  horizDelta = fragWorldPos.xz - casterPos.xz;
        float horizDist  = length(horizDelta);
        if (horizDist >= shadowRadius) continue;

        // Smooth circular falloff (dark center, fades at edges)
        float t = horizDist / shadowRadius;
        float shadow = 1.0 - smoothstep(0.0, 1.0, t);

        // Fade shadow with height (strong near floor, gone far below)
        float heightFade = 1.0 - smoothstep(0.0, 3.0, heightDiff);
        shadow *= heightFade;

        // Only shadow roughly upward-facing surfaces (floors, ramps)
        float upFacing = max(0.0, N.y);
        shadow *= upFacing;

        // Darken (0.5 = shadow darkness intensity, like Lithtech's SRCBLEND_ZERO/DESTBLEND_SRCCOLOR)
        shadowFactor = min(shadowFactor, 1.0 - shadow * 0.5);
    }
    totalLight *= shadowFactor;

    outColor = vec4(texColor.rgb * totalLight, texColor.a);

    // Linear distance fog (Lithtech-style: fog = mix(fogColor, litColor, factor))
    // fogParams.z == 1.0 means fog is enabled.  Sky mode skips fog via the early return above.
    if (lighting.fogParams.z > 0.5) {
        float dist = length(fragWorldPos - lighting.cameraPos.xyz);
        float fogNear = lighting.fogParams.x;
        float fogFar  = lighting.fogParams.y;
        float fogFactor = clamp((fogFar - dist) / (fogFar - fogNear), 0.0, 1.0);
        outColor.rgb = mix(lighting.fogColor.rgb, outColor.rgb, fogFactor);
    }
}
