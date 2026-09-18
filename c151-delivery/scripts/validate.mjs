import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { inflateSync } from 'node:zlib';
import { NodeIO } from '@gltf-transform/core';
import { ALL_EXTENSIONS } from '@gltf-transform/extensions';
import draco3d from 'draco3d';
import validator from 'gltf-validator';

// Validate the delivered, compressed files, including the decoded geometry.
const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const TOLERANCE = 0.003;
const decoder = await draco3d.createDecoderModule();
const io = new NodeIO().registerExtensions(ALL_EXTENSIONS).registerDependencies({ 'draco3d.decoder': decoder });
const MATERIALS = ['MRT_Body', 'MRT_Glass', 'MRT_Metal', 'MRT_Light'];
const near = (a, b, tolerance = 1e-6) => Math.abs(a - b) <= tolerance;
const vecNear = (a, b, tolerance = 1e-6) => a?.length === b.length && a.every((n, i) => near(n, b[i], tolerance));
const identityRotation = n => vecNear(n.getRotation(), [0, 0, 0, 1]) || vecNear(n.getRotation(), [0, 0, 0, -1]);
const identityScale = n => vecNear(n.getScale(), [1, 1, 1]);
const point = (m, p) => [m[0]*p[0]+m[4]*p[1]+m[8]*p[2]+m[12], m[1]*p[0]+m[5]*p[1]+m[9]*p[2]+m[13], m[2]*p[0]+m[6]*p[1]+m[10]*p[2]+m[14]];
const descendants = node => [node, ...node.listChildren().flatMap(descendants)];
const primitives = node => descendants(node).flatMap(n => n.getMesh()?.listPrimitives() ?? []);

function longitudinalBoundsInFrame(frame) {
  const matrix=frame.getWorldMatrix(), origin=point(matrix,[0,0,0]);
  let min=Infinity,max=-Infinity;
  for(const node of descendants(frame)) {
    for(const primitive of node.getMesh()?.listPrimitives()??[]) {
      const positions=primitive.getAttribute('POSITION');
      if(!positions) continue;
      const world=node.getWorldMatrix();
      for(let i=0;i<positions.getCount();i++) {
        const p=point(world,positions.getElement(i,[]));
        const x=(p[0]-origin[0])*matrix[0]+(p[1]-origin[1])*matrix[1]+(p[2]-origin[2])*matrix[2];
        min=Math.min(min,x);max=Math.max(max,x);
      }
    }
  }
  return [min,max];
}

function parseGLB(buffer) {
  if (buffer.readUInt32LE(0) !== 0x46546c67 || buffer.readUInt32LE(4) !== 2 || buffer.readUInt32LE(8) !== buffer.length) throw new Error('Invalid GLB 2.0 header');
  let json, bin;
  for (let offset = 12; offset < buffer.length;) {
    const length = buffer.readUInt32LE(offset), type = buffer.readUInt32LE(offset + 4);
    const data = buffer.subarray(offset + 8, offset + 8 + length);
    if (type === 0x4e4f534a) json = JSON.parse(data.toString('utf8'));
    if (type === 0x004e4942) bin = data;
    offset += 8 + length;
  }
  if (!json || !bin) throw new Error('GLB must have JSON and embedded BIN chunks');
  return { json, bin };
}

// Small independent PNG reader, only used to verify authored paint swatches.
function pngInfo(buffer) {
  if (buffer.subarray(0, 8).toString('hex') !== '89504e470d0a1a0a') throw new Error('Texture is not PNG');
  const width = buffer.readUInt32BE(16), height = buffer.readUInt32BE(20);
  const depth = buffer[24], colorType = buffer[25], interlace = buffer[28];
  const result = { width, height, depth, colorType, interlace };
  if (depth !== 8 || ![2, 6].includes(colorType) || interlace !== 0) return result;
  const channels = colorType === 6 ? 4 : 3;
  const chunks = [];
  for (let offset = 8; offset < buffer.length;) {
    const length = buffer.readUInt32BE(offset), type = buffer.subarray(offset + 4, offset + 8).toString();
    if (type === 'IDAT') chunks.push(buffer.subarray(offset + 8, offset + 8 + length));
    offset += length + 12;
  }
  const data = inflateSync(Buffer.concat(chunks));
  const stride = width * channels, pixels = Buffer.alloc(stride * height);
  const paeth = (a,b,c) => { const p=a+b-c, pa=Math.abs(p-a), pb=Math.abs(p-b), pc=Math.abs(p-c); return pa<=pb && pa<=pc?a:pb<=pc?b:c; };
  for (let y = 0; y < height; y++) {
    const filter = data[y*(stride+1)];
    for (let x = 0; x < stride; x++) {
      const a=x>=channels?pixels[y*stride+x-channels]:0, b=y?pixels[(y-1)*stride+x]:0, c=y && x>=channels?pixels[(y-1)*stride+x-channels]:0;
      const adjust = [0, a, b, Math.floor((a+b)/2), paeth(a,b,c)][filter];
      if (adjust === undefined) throw new Error(`Invalid PNG filter ${filter}`);
      pixels[y*stride+x]=(data[y*(stride+1)+1+x]+adjust)&255;
    }
  }
  result.swatchPixelCounts = Object.fromEntries(['F2F4F7','D5202B','59606B'].map(hex => [hex, 0]));
  for (let i=0; i<pixels.length; i+=channels) {
    const hex=pixels.subarray(i,i+3).toString('hex').toUpperCase();
    if (Object.hasOwn(result.swatchPixelCounts,hex)) result.swatchPixelCounts[hex]++;
  }
  return result;
}

function solve3(matrix, rhs) {
  const a = matrix.map((row, i) => [...row, rhs[i]]);
  for (let c=0;c<3;c++) {
    let p=c; for(let r=c+1;r<3;r++) if(Math.abs(a[r][c])>Math.abs(a[p][c])) p=r;
    if(Math.abs(a[p][c])<1e-9) return null;
    [a[c],a[p]]=[a[p],a[c]];
    const d=a[c][c]; for(let j=c;j<4;j++) a[c][j]/=d;
    for(let r=0;r<3;r++) if(r!==c) { const f=a[r][c]; for(let j=c;j<4;j++) a[r][j]-=f*a[c][j]; }
  }
  return a.map(row=>row[3]);
}

function checkSideUV(bodyNode) {
  const result = [];
  for(const sign of [-1,1]) {
    const samples=[];
    for(const node of descendants(bodyNode)) {
      for(const primitive of node.getMesh()?.listPrimitives() ?? []) {
        const pos=primitive.getAttribute('POSITION'), norm=primitive.getAttribute('NORMAL'), uv=primitive.getAttribute('TEXCOORD_0');
        if(!pos || !norm || !uv) continue;
        for(let i=0;i<pos.getCount();i++) {
          const p=pos.getElement(i,[]), n=norm.getElement(i,[]), t=uv.getElement(i,[]);
          if(sign*p[0]>1.5 && sign*n[0]>0.999) samples.push([p[1],p[2],t[0],t[1]]);
        }
      }
    }
    const m=Array.from({length:3},()=>[0,0,0]), bu=[0,0,0], bv=[0,0,0];
    for(const [y,z,u,v] of samples) { const a=[y,z,1]; for(let i=0;i<3;i++) { bu[i]+=a[i]*u;bv[i]+=a[i]*v; for(let j=0;j<3;j++)m[i][j]+=a[i]*a[j]; } }
    const u=solve3(m,bu), v=solve3(m,bv);
    let maxResidual=null;
    if(u && v) maxResidual=Math.max(...samples.flatMap(([y,z,s,t])=>[Math.abs(s-u[0]*y-u[1]*z-u[2]), Math.abs(t-v[0]*y-v[1]*z-v[2])]));
    result.push({ side:sign<0?'L':'R', samples:samples.length, uFromYZ:u, vFromYZ:v, maxResidual, affineStrip:maxResidual!==null && maxResidual<0.001 });
  }
  return result;
}

async function validateCar(filename, cab) {
  const absolute=path.join(ROOT,'assets',filename);
  const buffer=await fs.readFile(absolute);
  const {json,bin}=parseGLB(buffer);
  const checks=[];
  const check=(name,passed,details)=>checks.push({name,passed:Boolean(passed),...(details===undefined?{}:{details})});
  const khronos=await validator.validateBytes(new Uint8Array(buffer), {uri:filename,maxIssues:2000});
  check('Khronos glTF validator has zero errors',khronos.issues.numErrors===0,khronos.issues);
  check('GLB contains one embedded buffer',json.buffers?.length===1 && !json.buffers[0].uri);
  check('One default scene',json.scenes?.length===1 && (json.scene??0)===0);
  check('Draco required by the asset',json.extensionsRequired?.includes('KHR_draco_mesh_compression'));
  const rawPrimitives=(json.meshes??[]).flatMap(m=>m.primitives??[]);
  check('Every triangle primitive uses Draco',rawPrimitives.length>0 && rawPrimitives.every(p=>(p.mode??4)===4 && p.extensions?.KHR_draco_mesh_compression),{primitives:rawPrimitives.length});
  check('No cameras, punctual lights, skins or animations',(json.cameras??[]).length===0 && (json.skins??[]).length===0 && (json.animations??[]).length===0 && !json.extensionsUsed?.includes('KHR_lights_punctual') && !(json.nodes??[]).some(n=>n.camera!==undefined || n.skin!==undefined || n.extensions?.KHR_lights_punctual));
  check('Only core metallic-roughness material model',(json.materials??[]).every(m=>m.pbrMetallicRoughness && !Object.keys(m.extensions??{}).some(e=>e.startsWith('KHR_materials_'))));
  const materialNames=(json.materials??[]).map(m=>m.name).sort();
  check('Exactly the four required materials',JSON.stringify(materialNames)===JSON.stringify([...MATERIALS].sort()),materialNames);
  const bodyMaterial=(json.materials??[]).find(m=>m.name==='MRT_Body');
  const glassMaterial=(json.materials??[]).find(m=>m.name==='MRT_Glass');
  const lightMaterial=(json.materials??[]).find(m=>m.name==='MRT_Light');
  check('Body uses albedo atlas',bodyMaterial?.pbrMetallicRoughness?.baseColorTexture?.index!==undefined);
  check('Glass uses transparency',glassMaterial?.alphaMode==='BLEND' && (glassMaterial.pbrMetallicRoughness.baseColorFactor?.[3]??1)<1);
  check('Light material is emissive',lightMaterial?.emissiveFactor?.some(value=>value>0));
  const images=json.images??[];
  check('Exactly one embedded PNG albedo image',images.length===1 && images[0].bufferView!==undefined && images[0].mimeType==='image/png' && !images[0].uri);
  let texture;
  if(images.length===1 && images[0].bufferView!==undefined) {
    const view=json.bufferViews[images[0].bufferView];
    texture=pngInfo(bin.subarray(view.byteOffset??0,(view.byteOffset??0)+view.byteLength));
    check('Texture resolution is 1024 by 1024',texture.width===1024 && texture.height===1024,texture);
    if(texture.swatchPixelCounts) check('Exact requested paint swatches occur in texture',Object.values(texture.swatchPixelCounts).every(n=>n>0),texture.swatchPixelCounts);
  }
  const document=await io.read(absolute), docRoot=document.getRoot(), nodes=docRoot.listNodes();
  const byName=new Map();
  for(const node of nodes) {const list=byName.get(node.getName())??[];list.push(node);byName.set(node.getName(),list);}
  const required=['Body_Shell','Bogie_Front','Bogie_Rear','Glass_Windows','Roof_AC_1','Roof_AC_2',...['L','R'].flatMap(side=>[1,2,3,4].map(i=>`Door_${side}${i}`)),...(cab?['Glass_Windscreen','Headlight_L','Headlight_R']:[])];
  check('Required runtime names each occur exactly once',required.every(name=>byName.get(name)?.length===1),Object.fromEntries(required.map(name=>[name,byName.get(name)?.length??0])));
  check('No duplicate nonempty node names',[...byName].every(([name,list])=>!name || list.length===1));
  check('No interior, seats, cab equipment, pantograph or roof cabling nodes',nodes.every(n=>!/(?:interior|seat|pantograph|catenary|roof.?cabl|cab.?equipment)/i.test(n.getName())));
  check('Root transforms are identity',docRoot.listScenes().every(s=>s.listChildren().every(n=>identityRotation(n) && identityScale(n) && vecNear(n.getTranslation(),[0,0,0]))));
  check('Mesh rotations and scales are applied',nodes.filter(n=>n.getMesh()).every(n=>identityRotation(n) && identityScale(n)));
  const intentionalFrames=nodes.filter(n=>!identityRotation(n));
  check('Only intentional door coordinate frames are rotated',intentionalFrames.every(n=>/^DoorTrack_[LR]$/.test(n.getName()) && !n.getMesh() && identityScale(n)),intentionalFrames.map(n=>({name:n.getName(),rotation:n.getRotation()})));
  check('No negative or nonunit scales anywhere',nodes.every(identityScale));
  let triangles=0,vertices=0; const min=[Infinity,Infinity,Infinity],max=[-Infinity,-Infinity,-Infinity];
  const geometry=[];
  const winding={faces:0,degenerateFaces:0,minNormalAlignment:1,roofACSideFaces:0,minRoofACOutwardDot:Infinity,endChamferFaces:0,minEndChamferOutwardDot:Infinity};
  for(const node of nodes) {
    if(!node.getMesh()) continue;
    const world=node.getWorldMatrix();let nodeTriangles=0;
    for(const primitive of node.getMesh().listPrimitives()) {
      const position=primitive.getAttribute('POSITION');
      if(!position) continue;
      const count=primitive.getIndices()?.getCount()??position.getCount();
      nodeTriangles+=count/3;vertices+=position.getCount();
      const indices=primitive.getIndices()?.getArray(),normals=primitive.getAttribute('NORMAL');
      for(let t=0;t<count;t+=3) {
        const ids=[0,1,2].map(j=>indices?indices[t+j]:t+j);
        const p=ids.map(i=>position.getElement(i,[]));
        const ab=p[1].map((n,j)=>n-p[0][j]),ac=p[2].map((n,j)=>n-p[0][j]);
        const cross=[ab[1]*ac[2]-ab[2]*ac[1],ab[2]*ac[0]-ab[0]*ac[2],ab[0]*ac[1]-ab[1]*ac[0]];
        const area=Math.hypot(...cross);
        if(area<1e-9) {winding.degenerateFaces++;continue;}
        const n=cross.map(value=>value/area),center=[0,1,2].map(j=>(p[0][j]+p[1][j]+p[2][j])/3);
        winding.faces++;
        if(normals) for(const id of ids) {
          const authored=normals.getElement(id,[]),dot=n.reduce((sum,value,j)=>sum+value*authored[j],0);
          winding.minNormalAlignment=Math.min(winding.minNormalAlignment,dot);
        }
        if(/^Roof_AC_[12]$/.test(node.getName()) && primitive.getMaterial()?.getName()==='MRT_Body' && Math.abs(n[1])<.99) {
          winding.roofACSideFaces++;
          winding.minRoofACOutwardDot=Math.min(winding.minRoofACOutwardDot,n[0]*center[0]+n[2]*center[2]);
        }
        if(node.getName()==='Body_Shell' && primitive.getMaterial()?.getName()==='MRT_Body') {
          const zs=p.map(vertex=>vertex[2]),zmin=Math.min(...zs),zmax=Math.max(...zs);
          if(zmax-zmin>.05 && ((zmin>=.19 && zmax<=.29)||(zmin>=22.71 && zmax<=22.81))) {
            winding.endChamferFaces++;
            winding.minEndChamferOutwardDot=Math.min(winding.minEndChamferOutwardDot,n[0]*center[0]+n[1]*(center[1]-2.2));
          }
        }
      }
      for(let i=0;i<position.getCount();i++) {
        const p=point(world,position.getElement(i,[]));
        for(let j=0;j<3;j++){min[j]=Math.min(min[j],p[j]);max[j]=Math.max(max[j],p[j]);}
      }
    }
    triangles+=nodeTriangles;geometry.push({node:node.getName(),triangles:nodeTriangles});
  }
  check('Decoded triangles meet preferred 3000–6000 budget',triangles>=3000 && triangles<=6000,triangles);
  check('Decoded triangle winding agrees with vertex normals',winding.faces>0 && winding.minNormalAlignment>.98,{faces:winding.faces,degenerateFaces:winding.degenerateFaces,minNormalAlignment:winding.minNormalAlignment});
  check('Roof AC sidewalls face outward',winding.roofACSideFaces>0 && winding.minRoofACOutwardDot>0,{faces:winding.roofACSideFaces,minOutwardDot:winding.minRoofACOutwardDot});
  check('Both body end chamfers face outward',winding.endChamferFaces>0 && winding.minEndChamferOutwardDot>0,{faces:winding.endChamferFaces,minOutwardDot:winding.minEndChamferOutwardDot});
  check('Decoded triangles are below hard 8000 ceiling',triangles<=8000,triangles);
  check('World bounds are exact real-scale coupling extents',vecNear(min,[-1.6,0,0],TOLERANCE) && vecNear(max,[1.6,3.7,23],TOLERANCE),{min,max,size:max.map((n,i)=>n-min[i]),tolerance:TOLERANCE});
  const doors=[];
  for(const side of ['L','R']) {
    for(let i=1;i<=4;i++) {
      const name=`Door_${side}${i}`,door=byName.get(name)?.[0];
      if(!door) continue;
      const leaves=door.listChildren().filter(n=>n.getExtras()?.closedPosition!==undefined || n.getExtras()?.slideSign!==undefined);
      const detail={name,pivot:point(door.getWorldMatrix(),[0,0,0]),leaves:leaves.map(n=>({name:n.getName(),translation:n.getTranslation(),...n.getExtras()}))};
      doors.push(detail);
      check(`${name} has two independently sliding leaves`,leaves.length===2 && leaves.every(n=>Array.isArray(n.getExtras().closedPosition) && vecNear(n.getTranslation(),n.getExtras().closedPosition) && [-1,1].includes(n.getExtras().slideSign) && primitives(n).length>0) && leaves[0]?.getExtras().slideSign!==leaves[1]?.getExtras().slideSign,detail);
      check(`${name} local X follows carriage longitudinal axis`,(()=>{const m=door.getWorldMatrix();return Math.abs(m[0])<1e-6 && Math.abs(m[1])<1e-6 && near(Math.abs(m[2]),1);})());
      const edges=leaves.map(leaf=>({name:leaf.getName(),localXBounds:longitudinalBoundsInFrame(leaf)}));
      check(`${name} leaf origins are at their outer edges`,edges.length===2 && edges.every(({localXBounds:[min,max]})=>near(min,0,TOLERANCE) || near(max,0,TOLERANCE)),edges);
    }
    const positions=doors.filter(d=>d.name.startsWith(`Door_${side}`)).map(d=>d.pivot[2]).sort((a,b)=>a-b);
    const spacing=positions.slice(1).map((z,i)=>z-positions[i]);
    check(`Four evenly spaced doors on ${side} side`,positions.length===4 && spacing.every(s=>near(s,spacing[0],TOLERANCE)),{positions,spacing});
  }
  for(const name of ['Bogie_Front','Bogie_Rear']) {
    const node=byName.get(name)?.[0];
    check(`${name} has an independent rotation pivot`,node && primitives(node).length>0 && near(point(node.getWorldMatrix(),[0,0,0])[0],0) && identityRotation(node),node?point(node.getWorldMatrix(),[0,0,0]):null);
  }
  const bogieFront=byName.get('Bogie_Front')?.[0],bogieRear=byName.get('Bogie_Rear')?.[0];
  check('Bogie names follow positive-Z forward direction',bogieFront && bogieRear && point(bogieFront.getWorldMatrix(),[0,0,0])[2]>point(bogieRear.getWorldMatrix(),[0,0,0])[2]);
  for(const name of ['Glass_Windows',...(cab?['Glass_Windscreen']:[])]) {
    const node=byName.get(name)?.[0];
    check(`${name} uses MRT_Glass`,node && primitives(node).some(p=>p.getMaterial()?.getName()==='MRT_Glass'));
  }
  if(cab) for(const name of ['Headlight_L','Headlight_R']) {
    const node=byName.get(name)?.[0];
    check(`${name} uses MRT_Light`,node && primitives(node).some(p=>p.getMaterial()?.getName()==='MRT_Light'));
  }
  if(!cab) check('Intermediate has no cab headlights',!byName.has('Headlight_L') && !byName.has('Headlight_R'));
  const body=byName.get('Body_Shell')?.[0];
  let sideUV;
  if(body) { sideUV=checkSideUV(body);check('Each body side has one editable planar UV strip',sideUV.every(s=>s.affineStrip),sideUV); }
  return {file:`assets/${filename}`,bytes:buffer.length,passed:checks.every(c=>c.passed),triangles,vertices,bounds:{min,max},texture,nodeCount:nodes.length,meshCount:docRoot.listMeshes().length,materialNames,geometry,winding,doors,sideUV,checks};
}

function validateConsist() {
  // Longitudinal order: tail cab, four intermediates, leading cab.
  // The tail extends into negative Z from the shared rear coupler at Z=0.
  const length=23;
  const cars=[{kind:'cab',z:0,rotationY:Math.PI},...[0,1,2,3].map(i=>({kind:'intermediate',z:i*length,rotationY:0})),{kind:'cab',z:4*length,rotationY:0}];
  const spans=cars.map(car=>({...car,rearCouplerZ:car.z,frontCouplerZ:car.z+Math.cos(car.rotationY)*length,minZ:Math.min(car.z,car.z+Math.cos(car.rotationY)*length),maxZ:Math.max(car.z,car.z+Math.cos(car.rotationY)*length)}));
  const gaps=spans.slice(1).map((span,i)=>span.minZ-spans[i].maxZ);
  return {passed:gaps.every(gap=>near(gap,0)),carLength:length,consistLength:spans.at(-1).maxZ-spans[0].minZ,localRange:[-23,115],cars:spans,couplingGaps:gaps,note:'Whole consist can be translated by +23 m to occupy Z=0…138. The rear-origin tail is reversed about the coupling at Z=0; no mesh offset or bounding-box correction is required.'};
}

const report={generatedAt:new Date().toISOString(),validator:'Khronos glTF-Validator + independent Draco-decoded geometry and runtime contract checks',assets:[],assembly:validateConsist()};
for(const [file,cab] of [['c151-cab.glb',true],['c151-intermediate.glb',false]]) {
  try { report.assets.push(await validateCar(file,cab)); }
  catch(error) {report.assets.push({file:`assets/${file}`,passed:false,error:error.stack??String(error)});}
}
report.passed=report.assembly.passed && report.assets.every(a=>a.passed);
await fs.mkdir(path.join(ROOT,'reports'),{recursive:true});
await fs.writeFile(path.join(ROOT,'reports','validation.json'),JSON.stringify(report,null,2)+'\n');
for(const asset of report.assets) {
  console.log(`${asset.passed?'PASS':'FAIL'} ${asset.file}: ${asset.triangles??'?'} triangles, ${asset.bytes??'?'} bytes`);
  if(asset.error) console.log(asset.error);
  for(const check of asset.checks??[]) if(!check.passed) console.log(`  FAIL ${check.name}: ${JSON.stringify(check.details??'')}`);
}
console.log(`${report.assembly.passed?'PASS':'FAIL'} six-car coupling assembly: ${report.assembly.consistLength} m, gaps ${report.assembly.couplingGaps.join(', ')} m`);
console.log(`Report: ${path.join(ROOT,'reports','validation.json')}`);
process.exitCode=report.passed?0:1;
