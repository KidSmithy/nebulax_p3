import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const src=path.join(root,'node_modules/three'),dest=path.join(root,'vendor/three');
const seen=new Set();
async function copy(rel){
 if(seen.has(rel))return;seen.add(rel);
 const from=path.join(src,rel),to=path.join(dest,rel);
 await fs.mkdir(path.dirname(to),{recursive:true});await fs.copyFile(from,to);
 if(rel.endsWith('.js')){
  const code=await fs.readFile(from,'utf8');
  const refs=[...code.matchAll(/(?:from\s*|import\s*)['"]([^'"]+)['"]/g)].map(m=>m[1]);
  for(const ref of refs)if(ref.startsWith('.'))await copy(path.normalize(path.join(path.dirname(rel),ref)));
 }
}
for(const p of ['build/three.module.js','examples/jsm/controls/OrbitControls.js','examples/jsm/loaders/GLTFLoader.js','examples/jsm/loaders/DRACOLoader.js','examples/jsm/environments/RoomEnvironment.js','LICENSE'])await copy(p);
for(const p of ['draco_decoder.js','draco_decoder.wasm','draco_wasm_wrapper.js'])await copy('examples/jsm/libs/draco/gltf/'+p);
await fs.mkdir(path.join(root,'vendor/draco'),{recursive:true});
await fs.copyFile(path.join(src,'examples/jsm/libs/draco/README.md'),path.join(root,'vendor/draco/README.md'));
await fs.copyFile(path.join(root,'node_modules/sharp/LICENSE'),path.join(root,'vendor/draco/APACHE-2.0.txt'));
console.log(`Vendored ${seen.size} runtime files; no CDN required.`);
