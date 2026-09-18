import * as THREE from 'three';

export const CorrugationRibbonShaderMaterial = {
  uniforms: {
    u_depth_microns: { value: 12.0 },
    u_severity: { value: 0.25 },
    u_time: { value: 0.0 },
    u_speed: { value: 1.0 },
  },
  vertexShader: `
    uniform float u_depth_microns;
    uniform float u_severity;
    uniform float u_time;
    uniform float u_speed;

    varying vec2 vUv;
    varying vec3 vWorldPosition;
    varying float vDisplacement;

    void main() {
      vUv = uv;
      vec3 pos = position;

      // Corrugation displacement wave along track Z direction
      // 10-50 microns scaled for visual 3D Twin realism
      float waveFreq = 2.5;
      float wave = sin(pos.z * waveFreq + u_time * u_speed * 1.5);
      float displacement = (u_depth_microns / 80.0) * wave * 0.08;
      
      pos.y += displacement;
      vDisplacement = displacement;

      vec4 worldPosition = modelMatrix * vec4(pos, 1.0);
      vWorldPosition = worldPosition.xyz;
      gl_Position = projectionMatrix * viewMatrix * worldPosition;
    }
  `,
  fragmentShader: `
    uniform float u_severity;
    uniform float u_time;
    varying vec2 vUv;
    varying vec3 vWorldPosition;
    varying float vDisplacement;

    void main() {
      // Base polished steel rail color
      vec3 railColor = vec3(0.55, 0.58, 0.62);

      // Add metallic gradient reflection
      railColor += vec3(0.15) * pow(vUv.x, 2.0);

      // If high corrugation severity (> 0.70), project dynamic yellow/amber hazard alert stripes
      if (u_severity > 0.70) {
        float stripePattern = sin((vWorldPosition.z * 4.0) + (vUv.x * 6.0));
        if (stripePattern > 0.1) {
          vec3 amberAlert = vec3(0.96, 0.62, 0.05); // #f59e0b
          railColor = mix(railColor, amberAlert, 0.75);
        }
      } else if (u_severity > 0.40) {
        // Moderate corrugation tint
        railColor = mix(railColor, vec3(0.7, 0.5, 0.2), 0.35);
      }

      gl_FragColor = vec4(railColor, 1.0);
    }
  `,
};
