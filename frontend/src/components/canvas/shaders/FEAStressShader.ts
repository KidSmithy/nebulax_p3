import * as THREE from 'three';

export const FEAStressShaderMaterial = {
  uniforms: {
    u_stress_intensity: { value: 0.2 },
    u_hotspot_coords: { value: new THREE.Vector3(0.0, -0.4, 0.5) },
    u_time: { value: 0.0 },
    u_xray_opacity: { value: 1.0 },
  },
  vertexShader: `
    varying vec3 vPosition;
    varying vec3 vNormal;
    varying vec2 vUv;

    void main() {
      vPosition = position;
      vNormal = normalize(normalMatrix * normal);
      vUv = uv;
      gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
    }
  `,
  fragmentShader: `
    uniform float u_stress_intensity;
    uniform vec3 u_hotspot_coords;
    uniform float u_time;
    uniform float u_xray_opacity;

    varying vec3 vPosition;
    varying vec3 vNormal;
    varying vec2 vUv;

    // FEA Colormap: Navy Blue -> Emerald -> Amber -> Crimson/Magenta
    vec3 getStressColor(float val) {
      vec3 cNavy = vec3(0.118, 0.227, 0.541);    // #1e3a8a
      vec3 cEmerald = vec3(0.020, 0.588, 0.412); // #059669
      vec3 cAmber = vec3(0.851, 0.467, 0.024);   // #d97706
      vec3 cCrimson = vec3(0.882, 0.114, 0.282); // #e11d48

      if (val < 0.33) {
        return mix(cNavy, cEmerald, val / 0.33);
      } else if (val < 0.66) {
        return mix(cEmerald, cAmber, (val - 0.33) / 0.33);
      } else {
        return mix(cAmber, cCrimson, (val - 0.66) / 0.34);
      }
    }

    void main() {
      // Distance from critical weld node hotspot
      float dist = length(vPosition - u_hotspot_coords);
      float hotspot = exp(-dist * 2.8) * u_stress_intensity;
      float totalStress = clamp(0.12 + hotspot * 1.4 + 0.04 * sin(u_time * 8.0), 0.0, 1.0);

      vec3 baseColor = getStressColor(totalStress);

      // Add simple diffuse shading
      vec3 lightDir = normalize(vec3(1.0, 2.0, 1.5));
      float diff = max(dot(vNormal, lightDir), 0.25);
      vec3 finalColor = baseColor * (diff * 0.7 + 0.4);

      // Subtle contour fringe
      float contour = sin(totalStress * 31.415);
      if (abs(contour) > 0.85 && totalStress > 0.4) {
        finalColor += vec3(0.15);
      }

      gl_FragColor = vec4(finalColor, u_xray_opacity);
    }
  `,
};
