import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { DRACOLoader } from 'three/addons/loaders/DRACOLoader.js';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';

const viewport = document.getElementById('viewport');
const renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.setSize(viewport.clientWidth, viewport.clientHeight);
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.toneMapping = THREE.NeutralToneMapping;
renderer.toneMappingExposure = 1.0;
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
viewport.prepend(renderer.domElement);

const scene = new THREE.Scene();
scene.background = new THREE.Color('#e7e9e9');
scene.fog = new THREE.Fog('#e7e9e9', 170, 600);
const pmrem = new THREE.PMREMGenerator(renderer);
const room = new RoomEnvironment();
const environment = pmrem.fromScene(room, 0.06);
scene.environment = environment.texture;
scene.environmentIntensity = 0.55;
room.dispose();
pmrem.dispose();

const camera = new THREE.PerspectiveCamera(35, viewport.clientWidth / viewport.clientHeight, 0.1, 1500);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.075;
controls.minDistance = 7;
controls.maxDistance = 280;
controls.maxPolarAngle = Math.PI / 2 - 0.015;
controls.autoRotateSpeed = 0.45;

scene.add(new THREE.HemisphereLight(0xf8fcff, 0x677279, 1.4));
const key = new THREE.DirectionalLight(0xfff9ec, 2.2);
key.position.set(12, 24, 28);
key.castShadow = true;
key.shadow.mapSize.set(2048, 2048);
key.shadow.bias = -0.0003;
key.shadow.normalBias = 0.025;
key.shadow.radius = 3;
key.shadow.camera.near = 0.5;
key.shadow.camera.far = 230;
scene.add(key, key.target);
const fill = new THREE.DirectionalLight(0xecf4ff, 0.8);
fill.position.set(-15, 9, -20);
scene.add(fill);
const floor = new THREE.Mesh(new THREE.PlaneGeometry(2000, 2000), new THREE.MeshStandardMaterial({ color: '#cbd1d3', roughness: 0.92, metalness: 0 }));
floor.rotation.x = -Math.PI / 2;
floor.position.y = -0.18;
floor.receiveShadow = true;
scene.add(floor);

const query = new URLSearchParams(window.location.search);
const initialMode = ['cab', 'intermediate', 'consist'].includes(query.get('mode')) ? query.get('mode') : 'cab';
const initialView = ['perspective', 'side', 'front', 'roof'].includes(query.get('view')) ? query.get('view') : 'perspective';
let mode = initialMode;
let currentView = initialView;
let doorAmount = 0;
let modelGroup = new THREE.Group();
let trackGroup = new THREE.Group();
scene.add(modelGroup, trackGroup);
const models = {};
const state = window.previewState = { ready: false, mode, view: currentView, doors: 0, error: null, models: {} };
window.previewReady = false;

function addTrack(length) {
  trackGroup.traverse((node) => { if (node.isMesh) { node.geometry.dispose(); node.material.dispose(); } });
  scene.remove(trackGroup);
  trackGroup = new THREE.Group();
  const railMaterial = new THREE.MeshStandardMaterial({ color: '#7b858a', metalness: 0.75, roughness: 0.35 });
  const railLength = length + 10;
  for (const x of [-0.7175, 0.7175]) {
    const rail = new THREE.Mesh(new THREE.BoxGeometry(0.072, 0.12, railLength), railMaterial.clone());
    rail.position.set(x, -0.06, length / 2);
    rail.receiveShadow = true;
    trackGroup.add(rail);
  }
  railMaterial.dispose();
  const count = Math.ceil(railLength / 0.68);
  const sleepers = new THREE.InstancedMesh(new THREE.BoxGeometry(2.45, 0.055, 0.24), new THREE.MeshStandardMaterial({ color: '#a1aaae', roughness: 0.92 }), count);
  const matrix = new THREE.Matrix4();
  for (let i = 0; i < count; i++) sleepers.setMatrixAt(i, matrix.makeTranslation(0, -0.1475, -5 + i * 0.68));
  sleepers.receiveShadow = true;
  trackGroup.add(sleepers);
  scene.add(trackGroup);
}

function statsOf(root) {
  let triangles = 0;
  let meshCount = 0;
  const materialNames = new Set();
  root.traverse((node) => {
    if (!node.isMesh) return;
    meshCount++;
    triangles += (node.geometry.index?.count ?? node.geometry.attributes.position.count) / 3;
    for (const material of Array.isArray(node.material) ? node.material : [node.material]) materialNames.add(material.name);
  });
  root.updateMatrixWorld(true);
  const bounds = new THREE.Box3().setFromObject(root);
  return { triangles, meshCount, materials: [...materialNames].sort(), min: bounds.min.toArray(), max: bounds.max.toArray(), size: bounds.getSize(new THREE.Vector3()).toArray() };
}

function animateDoors(root, amount) {
  root.traverse((node) => {
    const { closedPosition, slideSign } = node.userData;
    if (Array.isArray(closedPosition) && typeof slideSign === 'number') {
      node.position.x = closedPosition[0] + slideSign * 0.77 * amount;
    }
  });
}

function setDoors(value) {
  doorAmount = THREE.MathUtils.clamp(Number(value), 0, 1);
  animateDoors(modelGroup, doorAmount);
  document.getElementById('doors').value = String(doorAmount);
  document.getElementById('door-value').textContent = doorAmount === 0 ? 'Closed' : doorAmount === 1 ? 'Open' : `${Math.round(doorAmount * 100)}%`;
  state.doors = doorAmount;
  modelGroup.updateMatrixWorld(true);
}

function setView(view) {
  if (!['perspective', 'side', 'front', 'roof'].includes(view)) throw new Error(`Unknown view: ${view}`);
  currentView = view;
  state.view = view;
  const length = mode === 'consist' ? 138 : 23;
  const aspect = viewport.clientWidth / viewport.clientHeight;
  const target = new THREE.Vector3(0, 1.65, length / 2);
  const wide = mode === 'consist';
  // Frame the actual model envelope, reserving the lower area for controls.
  // A negative-X camera puts the forward cab at the right of the composition.
  const direction = new THREE.Vector3(...(view === 'side' ? [-1, 0.045, 0] : view === 'front' ? [0, 0.018, 1] : view === 'roof' ? [-0.01, 1, 0.001] : [-1, wide ? 0.25 : 0.34, wide ? 0.64 : 0.85])).normalize();
  camera.position.copy(target).add(direction);
  camera.lookAt(target);
  const inverseRotation = camera.quaternion.clone().invert();
  const tanY = Math.tan(THREE.MathUtils.degToRad(camera.fov / 2));
  const tanX = tanY * aspect;
  let distance = 0;
  for (const x of [-1.6, 1.6]) for (const y of [0, 3.7]) for (const z of [0, length]) {
    const corner = new THREE.Vector3(x, y, z).sub(target).applyQuaternion(inverseRotation);
    distance = Math.max(distance, Math.abs(corner.x) / (tanX * 0.87) + corner.z, Math.abs(corner.y) / (tanY * 0.43) + corner.z);
  }
  distance *= 1.045;
  controls.target.copy(target);
  camera.position.copy(target).addScaledVector(direction, distance);
  camera.setViewOffset(viewport.clientWidth, viewport.clientHeight, 0, viewport.clientHeight * 0.055, viewport.clientWidth, viewport.clientHeight);
  camera.lookAt(target);
  controls.update();
  for (const button of document.querySelectorAll('[data-view]')) button.classList.toggle('selected', button.dataset.view === view);
}

function setMode(nextMode) {
  if (!models.cab || !models.intermediate) return;
  if (!['cab', 'intermediate', 'consist'].includes(nextMode)) throw new Error(`Unknown model mode: ${nextMode}`);
  mode = nextMode;
  state.mode = mode;
  scene.remove(modelGroup);
  modelGroup = new THREE.Group();
  modelGroup.name = 'Preview_Models';
  if (mode === 'consist') {
    const rearCab = models.cab.clone(true);
    rearCab.rotation.y = Math.PI;
    rearCab.position.z = 23;
    rearCab.name = 'Car_1_Cab_Reversed';
    modelGroup.add(rearCab);
    for (let i = 1; i <= 4; i++) {
      const car = models.intermediate.clone(true);
      car.position.z = i * 23;
      car.name = `Car_${i + 1}_Intermediate`;
      modelGroup.add(car);
    }
    const leadingCab = models.cab.clone(true);
    leadingCab.position.z = 115;
    leadingCab.name = 'Car_6_Cab';
    modelGroup.add(leadingCab);
  } else modelGroup.add(models[mode].clone(true));
  scene.add(modelGroup);
  setDoors(doorAmount);
  const length = mode === 'consist' ? 138 : 23;
  addTrack(length);
  key.target.position.set(0, 0, length / 2);
  key.position.set(18, mode === 'consist' ? 72 : 24, length / 2 + 18);
  const span = mode === 'consist' ? 84 : 22;
  Object.assign(key.shadow.camera, { left: -span, right: span, top: span, bottom: -span });
  key.shadow.camera.updateProjectionMatrix();
  const stats = statsOf(modelGroup);
  state.current = stats;
  state.modelGroup = modelGroup;
  document.getElementById('model-label').textContent = mode === 'cab' ? 'DRIVING CAR' : mode === 'intermediate' ? 'INTERMEDIATE CAR' : 'SIX-CAR CONSIST';
  document.getElementById('dimension-label').textContent = `${length.toFixed(1)} × 3.2 × 3.7`;
  document.getElementById('triangle-label').textContent = Math.round(stats.triangles).toLocaleString('en-US');
  document.getElementById('material-label').textContent = '4';
  for (const button of document.querySelectorAll('[data-mode]')) {
    const selected = button.dataset.mode === mode;
    button.classList.toggle('selected', selected);
    button.setAttribute('aria-pressed', String(selected));
  }
  setView(currentView);
}

const draco = new DRACOLoader();
draco.setDecoderPath('/vendor/three/examples/jsm/libs/draco/gltf/');
const loader = new GLTFLoader().setDRACOLoader(draco);
window.previewLoadPromise = Promise.all([
  loader.loadAsync('/assets/c151-cab.glb'),
  loader.loadAsync('/assets/c151-intermediate.glb')
]).then(([cab, intermediate]) => {
  for (const [name, gltf] of Object.entries({ cab, intermediate })) {
    models[name] = gltf.scene;
    models[name].traverse((node) => {
      if (!node.isMesh) return;
      node.castShadow = true;
      node.receiveShadow = true;
    });
    state.models[name] = statsOf(models[name]);
  }
  setMode(initialMode);
  if (query.has('doors')) setDoors(Number(query.get('doors')) || 0);
  document.getElementById('loading').classList.add('hidden');
  state.ready = true;
  window.previewReady = true;
  renderer.render(scene, camera);
  return state;
}).catch((error) => {
  state.error = String(error);
  const loading = document.getElementById('loading');
  loading.classList.add('error');
  loading.textContent = 'Could not load the models. Serve the c151-delivery folder over HTTP, then open /preview/.';
  console.error(error);
});

window.setPreviewMode = setMode;
window.setPreviewView = setView;
window.setDoorsOpen = setDoors;
window.previewRenderer = renderer;
window.previewScene = scene;
window.previewCamera = camera;
window.previewControls = controls;
async function downloadPNG() {
  renderer.render(scene, camera);
  const png=renderer.domElement.toDataURL('image/png');
  const name=`c151-${mode}-${currentView}${doorAmount > 0 ? '-doors-open' : ''}.png`;
  try {
    const result=await fetch('/__capture',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({name,png})});
    if(result.ok){document.getElementById('download-png').textContent='PNG saved ✓';return;}
  } catch {}
  const link = document.createElement('a');
  link.href = png;
  link.download = name;
  link.click();
}
window.downloadPreviewPNG = downloadPNG;
document.getElementById('download-png').addEventListener('click', downloadPNG);
for (const button of document.querySelectorAll('[data-mode]')) button.addEventListener('click', () => setMode(button.dataset.mode));
for (const button of document.querySelectorAll('[data-view]')) button.addEventListener('click', () => setView(button.dataset.view));
document.getElementById('doors').addEventListener('input', (event) => setDoors(event.target.value));
document.getElementById('autorotate').addEventListener('change', (event) => { controls.autoRotate = event.target.checked; });
document.getElementById('wireframe').addEventListener('change', (event) => {
  for (const root of Object.values(models)) root.traverse((node) => {
    if (node.isMesh) for (const material of Array.isArray(node.material) ? node.material : [node.material]) material.wireframe = event.target.checked;
  });
  state.wireframe = event.target.checked;
});
window.addEventListener('resize', () => {
  camera.aspect = viewport.clientWidth / viewport.clientHeight;
  renderer.setSize(viewport.clientWidth, viewport.clientHeight);
  camera.updateProjectionMatrix();
  setView(currentView);
});
renderer.setAnimationLoop(() => { controls.update(); renderer.render(scene, camera); });
