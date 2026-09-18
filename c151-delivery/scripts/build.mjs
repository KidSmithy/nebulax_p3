import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { Document, NodeIO } from '@gltf-transform/core';
import { KHRDracoMeshCompression } from '@gltf-transform/extensions';
import { draco } from '@gltf-transform/functions';
import draco3d from 'draco3d';
import sharp from 'sharp';
import { ShapeUtils, Vector2 } from 'three';

const DIR=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const OUT=path.join(DIR,'assets');
await fs.mkdir(OUT,{recursive:true});
const C={white:'#F2F4F7',red:'#D5202B',grey:'#59606B',black:'#171E26',metal:'#555F68',silver:'#AEB7BF',rubber:'#222830',glass:'#101B23',light:'#FFF2CD',marker:'#651E25'};
const SW={white:32,grey:100,black:168,metal:236,silver:304,rubber:372,glass:440,light:508,marker:576};
const sw=c=>[(SW[c]+20)/1024,970/1024];
const sideY=s=>s===-1?16:188;
const sideUV=(s,z,y)=>[(16+(z-.28)/22.44*992)/1024,(sideY(s)+(3.22-y)/2.28*152)/1024];
const frontUV=(x,y,rear=false)=>[((rear?272:16)+(x+1.6)/3.2*240)/1024,(488+(3.5-y)/2.56*240)/1024];
const roofUV=(x,z)=>[(16+(z-.28)/22.44*992)/1024,(358+(x+1.6)/3.2*112)/1024];
const acUV=(x,z,center)=>[(540+(x+1.12)/2.24*452)/1024,(492+(z-center+1.55)/3.10*222)/1024];
const doorCenters=[3.7,8.7,13.7,18.7];
const roundRect=(u0,v0,u1,v1,r=.05,n=3)=>{
 const pts=[];
 for(const [cx,cy,start] of [[u1-r,v0+r,-Math.PI/2],[u1-r,v1-r,0],[u0+r,v1-r,Math.PI/2],[u0+r,v0+r,Math.PI]])
  for(let i=0;i<=n;i++){let a=start+i/n*Math.PI/2;pts.push([cx+Math.cos(a)*r,cy+Math.sin(a)*r]);}
 return pts;
};
const rect=(a,b,c,d)=>[[a,b],[c,b],[c,d],[a,d]];
const windowRects=cab=>{
 let a=[[.9,2.15,2.63,2.98]];
 for(let i=0;i<3;i++) {let start=doorCenters[i]+1.07,end=doorCenters[i+1]-1.07,m=(start+end)/2;a.push([start,2.15,m-.045,2.98],[m+.045,2.15,end,2.98]);}
 if(cab)a.push([20.05,2.14,21.03,2.98],[21.48,2.13,22.32,3.04]);
 else a.push([19.77,2.15,21.05,2.98],[21.14,2.15,22.32,2.98]);
 return a;
};

async function atlas(cab){
 const e=[];
 const r=(x,y,w,h,fill,stroke='',sw=1,rad=0)=>e.push(`<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="${rad}" fill="${fill}" ${stroke?`stroke="${stroke}" stroke-width="${sw}"`:''}/>`);
 const ln=(x1,y1,x2,y2,c,w=1)=>e.push(`<path d="M${x1} ${y1}L${x2} ${y2}" stroke="${c}" stroke-width="${w}"/>`);
 r(0,0,1024,1024,C.white);
 for(const s of [-1,1]){
  const xy=(z,y)=>sideUV(s,z,y).map(v=>v*1024);
  const span=(z0,y0,z1,y1,color)=>{let [x,yt]=xy(z0,y1),[xr,yb]=xy(z1,y0);r(x,yt,xr-x,yb-yt,color);};
  span(.15,.9,22.85,3.26,C.white);
  span(.15,.9,22.85,1.23,C.grey);
  span(.15,1.69,22.85,2.035,C.red);
  span(.15,2.08,22.85,3.08,C.black);
  for(let z=1.3;z<22;z+=2.25){let [x,y]=xy(z,1.66),[_,yb]=xy(z,1.27);ln(x,y,x,yb,'#C9CED3',.65);}
  for(const dc of doorCenters){
   for(const z of [dc-.75,dc,dc+.75]){let [x,y]=xy(z,3.04),[_,yb]=xy(z,1.1);ln(x,y,x,yb,C.black,1.25);}
   for(const [a,b] of [[dc-.68,dc-.07],[dc+.07,dc+.68]]){
    // Painted silver window frame and white door upper surround.
    span(a-.03,2.055,b+.03,3.025,'#D7DCE1');
    const [x,y]=xy(a,2.985),[xr,yb]=xy(b,2.145);r(x,y,xr-x,yb-y,C.black,'#78828B',1,3);
   }
  }
  for(const q of windowRects(cab)){
   let [x,y]=xy(q[0]-.035,q[3]+.035),[xr,yb]=xy(q[2]+.035,q[1]-.035);r(x,y,xr-x,yb-y,'#111820','#6C747C',.8,3);
  }
  for(const y of [1.245,3.15]){let [x,yp]=xy(.28,y),[xr]=xy(22.72,y);ln(x,yp,xr,yp,'#A6AEB7',.7);}
 }
 r(8,351,1008,126,'#D8DDE1');
 for(const z of [.7,3,8.6,11.5,14.4,20,22.2]){let [x]=roofUV(0,z);ln(x*1024,358,x*1024,470,'#ADB5BC',1);}
 for(const rear of [false,true]){
  const p=(x,y)=>frontUV(x,y,rear).map(v=>v*1024);
  const block=(x0,y0,x1,y1,col)=>{let [x,y]=p(x0,y1),[xr,yb]=p(x1,y0);r(x,y,xr-x,yb-y,col);};
  block(-1.66,.92,1.66,3.55,C.white);block(-1.66,.92,1.66,1.26,C.grey);
  block(-1.66,1.69,1.66,2.035,C.red);
  if(!rear&&cab){
   let [x,y]=p(-1.4,3.23),[xr,yb]=p(1.4,2.05);r(x,y,xr-x,yb-y,C.black,'',0,15);
   block(-1.32,3.00,1.32,3.22,'#090F16');
   // Intentionally blank full-width destination panel; no wordmarks.
   let [a,b]=p(-1.2,3.025),[c,d]=p(1.2,3.195);ln(a,b,c,b,'#28343C',1);
   for(const x of [-.61,.61]){let [u,v]=p(x,1.32),[u2,v2]=p(x,1.64);ln(u,v,u2,v2,'#9BA5AF',.65);}
  } else{
   block(-.57,1.12,.57,3.12,'#313A43');
   for(const x of [-.69,.69]){let [u,v]=p(x,1.05),[u2,v2]=p(x,3.17);ln(u,v,u2,v2,'#2B333B',3);}
  }
 }
 r(532,484,468,238,'#D0D6DC','#7F8B94',2,8);
 for(let y=503;y<704;y+=7){ln(557,y,697,y,'#727F89',3);ln(842,y,975,y,'#727F89',3);}
 for(const y of [550,651]){
  e.push(`<circle cx="771" cy="${y}" r="43" fill="#7A858F" stroke="#AAB2BA" stroke-width="4"/>`);
  for(let j=-36;j<=36;j+=6){let half=Math.sqrt(38*38-j*j);ln(771-half,y+j,771+half,y+j,'#3C4854',2);}
 }
 for(const [k,x] of Object.entries(SW))r(x,936,48,76,C[k]);
 const svg=`<svg xmlns="http://www.w3.org/2000/svg" width="1024" height="1024">${e.join('')}</svg>`;
 await fs.writeFile(path.join(OUT,`c151-${cab?'cab':'intermediate'}-albedo.svg`),svg);
 const buf=await sharp(Buffer.from(svg)).png().toBuffer();
 await fs.writeFile(path.join(OUT,`c151-${cab?'cab':'intermediate'}-albedo.png`),buf);
 return buf;
}

class Geo{
 constructor(){this.p=[];this.n=[];this.uv=[];this.i=[];}
 tri(a,b,c,uvs,color='white',normal=null){
  let ab=b.map((x,i)=>x-a[i]),ac=c.map((x,i)=>x-a[i]);let n=normal||[ab[1]*ac[2]-ab[2]*ac[1],ab[2]*ac[0]-ab[0]*ac[2],ab[0]*ac[1]-ab[1]*ac[0]];
  let l=Math.hypot(...n);if(l<1e-9)return;n=n.map(x=>x/l);
  let first=this.p.length/3;
  for(let j=0;j<3;j++){this.p.push(...[a,b,c][j]);this.n.push(...n);this.uv.push(...(uvs?.[j]||sw(color)));this.i.push(first+j);}
 }
 quad(a,b,c,d,uvs,color='white',normal=null){this.tri(a,b,c,uvs?[uvs[0],uvs[1],uvs[2]]:null,color,normal);this.tri(a,c,d,uvs?[uvs[0],uvs[2],uvs[3]]:null,color,normal);}
 plane(poly,map,uvMap,color='white',outward){
  let tris=ShapeUtils.triangulateShape(poly.map(q=>new Vector2(...q)),[]);
  for(let t of tris){let pts=t.map(i=>map(...poly[i])),uv=t.map(i=>uvMap?uvMap(...poly[i]):sw(color));
   if(outward){let a=pts[1].map((x,i)=>x-pts[0][i]),b=pts[2].map((x,i)=>x-pts[0][i]);let n=[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];if(n.reduce((s,x,i)=>s+x*outward[i],0)<0){[pts[1],pts[2]]=[pts[2],pts[1]];[uv[1],uv[2]]=[uv[2],uv[1]];}}
   this.tri(...pts,uv,color);
  }
 }
 panel(outer,holes,map,uvMap,outward){
  let all=[...outer,...holes.flat()],tris=ShapeUtils.triangulateShape(outer.map(q=>new Vector2(...q)),holes.map(h=>h.map(q=>new Vector2(...q))));
  for(let t of tris){let pts=t.map(i=>map(...all[i])),uv=t.map(i=>uvMap(...all[i]));let a=pts[1].map((x,i)=>x-pts[0][i]),b=pts[2].map((x,i)=>x-pts[0][i]);let n=[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];
   if(n.reduce((s,x,i)=>s+x*outward[i],0)<0){[pts[1],pts[2]]=[pts[2],pts[1]];[uv[1],uv[2]]=[uv[2],uv[1]];}this.tri(...pts,uv);
  }
 }
 ring(a,b,mapA,mapB,color='black',reverse=false){
  for(let i=0;i<a.length;i++){let j=(i+1)%a.length;let q=[mapA(...a[i]),mapA(...a[j]),mapB(...b[j]),mapB(...b[i])];if(reverse)q.reverse();this.quad(...q,null,color);}
 }
 box(x0,y0,z0,x1,y1,z1,color='metal',topUV=null){
  const vs=[[x0,y0,z0],[x1,y0,z0],[x1,y1,z0],[x0,y1,z0],[x0,y0,z1],[x1,y0,z1],[x1,y1,z1],[x0,y1,z1]];
  for(const f of [[0,3,2,1],[4,5,6,7],[0,4,7,3],[1,2,6,5],[0,1,5,4],[3,7,6,2]]){let pts=f.map(i=>vs[i]);let uv=(topUV&&f[0]===3)?pts.map(p=>topUV(p[0],p[2])):null;this.quad(...pts,uv,color);}
 }
 cylinder(axis,center,radius,length,segs=16,color='metal'){
  let axes=axis===0?[1,2]:axis===1?[2,0]:[0,1];
  const p=(angle,d,r=radius)=>{let q=[...center];q[axis]+=d;q[axes[0]]+=Math.cos(angle)*r;q[axes[1]]+=Math.sin(angle)*r;return q;};
  for(let j=0;j<segs;j++){let a=j/segs*Math.PI*2,b=(j+1)/segs*Math.PI*2;this.quad(p(a,-length/2),p(b,-length/2),p(b,length/2),p(a,length/2),null,color);this.tri(p(0,-length/2,0),p(b,-length/2),p(a,-length/2),null,color);this.tri(p(0,length/2,0),p(a,length/2),p(b,length/2),null,color);}
 }
 translate(v){this.p=this.p.map((x,i)=>x-v[i%3]);return this;}
 // Convert global geometry into the unrotated door mesh's X/Y/Z frame.
 doorLocal(origin){let p=[];for(let i=0;i<this.p.length;i+=3){let [x,y,z]=this.p.slice(i,i+3);p.push(-(z-origin[2]),y-origin[1],x-origin[0]);}this.p=p;let n=[];for(let i=0;i<this.n.length;i+=3)n.push(-this.n[i+2],this.n[i+1],this.n[i]);this.n=n;return this;}
}

async function build(cab){
 const kind=cab?'cab':'intermediate',doc=new Document(),buffer=doc.createBuffer();
 const texture=doc.createTexture(`C151_${kind}_Albedo`).setImage(await atlas(cab)).setMimeType('image/png');
 const mats={};
 for(const name of ['Body','Glass','Metal','Light']){
  let m=doc.createMaterial(`MRT_${name}`).setBaseColorTexture(texture).setMetallicFactor(name==='Metal'?.7:name==='Glass'?.12:0).setRoughnessFactor(name==='Metal'?.57:name==='Glass'?.18:.63);
  if(name==='Glass')m.setAlphaMode('BLEND').setBaseColorFactor([1,1,1,.97]).setDoubleSided(true);
  if(name==='Light')m.setEmissiveFactor([1,.84,.58]).setRoughnessFactor(.28);
  mats[name]=m;
 }
 const scene=doc.createScene(`C151_${kind}`),root=doc.createNode(`C151_${cab?'Cab':'Intermediate'}`).setExtras({units:'metres',forward:'+Z',up:'+Y',couplingLength:23,rearCoupling:[0,0,0],frontCoupling:[0,0,23],exteriorOnly:true});scene.addChild(root);
 let metrics=[];
 function node(name,geos,parent=root,translation=[0,0,0],extras={}){
  let nd=doc.createNode(name).setTranslation(translation).setExtras(extras);parent.addChild(nd);
  let nonempty=Object.entries(geos).filter(([_,g])=>g.i.length);
  if(nonempty.length){let mesh=doc.createMesh(name);
   for(const [mat,g] of nonempty){let prim=doc.createPrimitive().setMaterial(mats[mat]);
    for(const [sem,type,array] of [['POSITION','VEC3',new Float32Array(g.p)],['NORMAL','VEC3',new Float32Array(g.n)],['TEXCOORD_0','VEC2',new Float32Array(g.uv)]])prim.setAttribute(sem,doc.createAccessor().setType(type).setArray(array).setBuffer(buffer));
    prim.setIndices(doc.createAccessor().setType('SCALAR').setArray(new Uint16Array(g.i)).setBuffer(buffer));mesh.addPrimitive(prim);metrics.push({node:name,material:mat,triangles:g.i.length/3});
   }nd.setMesh(mesh);
  }return nd;
 }
 const body=new Geo(),metal=new Geo(),windows=new Geo(),windscreen=new Geo();
 const openings=[];
 for(const s of [-1,1]){
  let holes=[],wr=windowRects(cab);
  for(const dc of doorCenters)holes.push(roundRect(dc-.77,1.095,dc+.77,3.06,.045,2));
  for(const q of wr)holes.push(roundRect(...q,.055,2));
  body.panel(rect(.28,.94,22.72,3.22),holes,(z,y)=>[s*1.6,y,z],(z,y)=>sideUV(s,z,y),[s,0,0]);
  for(let i=0;i<holes.length;i++){
   let h=holes[i];metal.ring(h,h,(z,y)=>[s*1.6,y,z],(z,y)=>[s*1.566,y,z],'black',s===1);
   if(i>=4){windows.plane(h,(z,y)=>[s*1.595,y,z],null,'glass',[s,0,0]);}
  }
  // Lower bevel returns, upper roof shoulder and sill folded sheet.
  const profile=s===1?[[1.6,3.22],[1.59,3.32],[1.55,3.40],[1.46,3.46],[1.32,3.49]]:[[-1.32,3.49],[-1.46,3.46],[-1.55,3.40],[-1.59,3.32],[-1.6,3.22]];
  for(let j=0;j<profile.length-1;j++){let [x,y]=profile[j],[xx,yy]=profile[j+1];body.quad([xx,yy,.28],[xx,yy,22.72],[x,y,22.72],[x,y,.28],null,'white');}
  let sill=[[s*1.6,.94,.28],[s*1.49,.84,.28],[s*1.49,.84,22.72],[s*1.6,.94,22.72]];if(s>0)sill.reverse();body.quad(...sill,null,'grey');
 }
 body.quad([-1.32,3.49,.28],[-1.32,3.49,22.72],[1.32,3.49,22.72],[1.32,3.49,.28],[roofUV(-1.32,.28),roofUV(-1.32,22.72),roofUV(1.32,22.72),roofUV(1.32,.28)]);
 body.box(-1.49,.83,.28,1.49,.94,22.72,'grey');
 // Rounded corner extrusion keeps the front a near-vertical slab.
 const silhouette=[[-1.49,.94],[1.49,.94],[1.6,1.10],[1.6,3.22],[1.59,3.32],[1.55,3.40],[1.46,3.46],[1.32,3.49],[-1.32,3.49],[-1.46,3.46],[-1.55,3.40],[-1.59,3.32],[-1.6,3.22],[-1.6,1.10]];
 const insetSil=silhouette.map(([x,y])=>[x*.965,y>3.2?y-.02:y]);
 for(const [z,end] of [[.28,-1],[22.72,1]]){
  let faceZ=z+end*.08,isCab=cab&&end===1;
  body.ring(silhouette,insetSil,(x,y)=>[x,y,z],(x,y)=>[x,y,faceZ],'white',end===-1);
  let holes=isCab?[roundRect(-1.285,2.12,-.085,2.94,.09,4),roundRect(.085,2.12,1.285,2.94,.09,4)]:[];
  body.panel(insetSil,holes,(x,y)=>[x,y,faceZ],(x,y)=>frontUV(x,y,!isCab),[0,0,end]);
  if(isCab){
   for(const h of holes){metal.ring(h,h,(x,y)=>[x,y,faceZ+.001],(x,y)=>[x,y,faceZ-.023],'black');windscreen.plane(h,(x,y)=>[x,y,faceZ-.012],null,'glass',[0,0,1]);}
   // Thin physical windshield wipers, not cab equipment.
   for(const sx of [-.72,.72]){metal.box(sx-.013,2.135,faceZ+.016,sx+.013,2.48,faceZ+.033,'black');metal.box(sx-.035,2.33,faceZ+.036,sx+.035,2.62,faceZ+.048,'rubber');}
   for(const [s,label] of [[-1,'L'],[1,'R']]){
    let x=s*1.02,lamp=new Geo();metal.box(x-.27,1.40,faceZ+.006,x+.27,1.66,faceZ+.046,'silver');metal.box(x-.245,1.42,faceZ+.047,x+.245,1.64,faceZ+.055,'black');
    lamp.cylinder(2,[x-s*.11,1.53,faceZ+.067],.080,.022,16,'light');metal.cylinder(2,[x+s*.11,1.53,faceZ+.067],.077,.022,16,'marker');
    node(`Headlight_${label}`,{Light:lamp.translate([x,1.53,faceZ+.067])},root,[x,1.53,faceZ+.067]);
   }
  }else{
   // Shallow gangway bellows, exterior only, leaves coupling face unobstructed.
   const gang=roundRect(-.70,1.13,.70,3.16,.06,2);
   for(let j=0;j<4;j++){let a=z+end*(.085+j*.035),b=a+end*.015;let inner=roundRect(-.60,1.23,.60,3.06,.045,2);metal.ring(gang,inner,(x,y)=>[x,y,a],(x,y)=>[x,y,b],j%2?'rubber':'black',end===-1);}
   metal.box(-.45,1.10,Math.min(z+end*.1,z+end*.18),.45,1.15,Math.max(z+end*.1,z+end*.18),'silver');
  }
 }
 // Coupler faces at Z=0 and Z=23 define the authored length and origin.
 for(const end of [0,23]){
  let a=end===0?0:22.57,b=end===0?.43:23;
  metal.box(-.135,.39,a,.135,.60,b,'metal');metal.box(-.225,.35,end===0?0:22.88,.225,.65,end===0?.12:23,'metal');
  metal.box(-.11,.42,end===0?-.0:22.99,.11,.57,end===0?.011:23,'black');
 }
 // Equipment skirt boxes, battery cases, third-rail gear (no roof cabling).
 for(const [a,b,h] of [[6.0,8.2,.54],[8.45,11.05,.47],[11.3,14.5,.56],[14.8,16.95,.50]]){
  metal.box(-1.29,h,a,1.29,.83,b,'grey');
  for(const s of [-1,1])for(let z=a+.17;z<b-.1;z+=.22){let q=[[s*1.293,h+.1,z],[s*1.293,.78,z],[s*1.293,.78,z+.06],[s*1.293,h+.1,z+.06]];if(s<0)q.reverse();metal.quad(...q,null,'black');}
 }
 node('Body_Shell',{Body:body,Metal:metal},root,[0,0,0],{sideUV:{left:{pixels:[16,16,1008,168]},right:{pixels:[16,188,1008,340]}},bandY:[1.69,2.035],bandIsTextureOnly:true});
 node('Glass_Windows',{Glass:windows});
 node('Glass_Windscreen',{Glass:windscreen});

 // Door track frames explicitly encode a local X axis along the carriage.
 for(const [s,label] of [[-1,'L'],[1,'R']]){
  let frame=node(`DoorTrack_${label}`,{});frame.setRotation([0,Math.SQRT1_2,0,Math.SQRT1_2]);frame.setExtras({purpose:'Local X slide basis; intentional non-mesh coordinate frame',localXInCarriage:[0,0,-1]});
  for(let i=0;i<4;i++){
   let dc=doorCenters[i],outer=[s*1.588,1.1,dc+.748],cp=[-outer[2],outer[1],outer[0]];
   let ctrl=node(`Door_${label}${i+1}`,{},frame,cp,{pair:true,doorNumbering:'rear to front',slideAxis:'local X',stroke:.77,origin:'pair outer longitudinal edge'});
   // Leaf_A occupies forward half; Leaf_B rear half. Both origins at own outer edge.
   for(const [leaf,z0,z1,sign] of [['A',dc+.009,dc+.746,-1],['B',dc-.746,dc-.009,1]]){
    const edge=leaf==='A'?z1:z0,origin=[s*1.588,1.1,edge];
    let g=new Geo(),gg=new Geo(),gm=new Geo();let h=roundRect(z0+.08,2.145,z1-.08,2.985,.065,2),panel=roundRect(z0,1.106,z1,3.048,.025,2);
    g.panel(panel,[h],(z,y)=>[s*1.589,y,z],(z,y)=>sideUV(s,z,y),[s,0,0]);
    gm.ring(panel,panel,(z,y)=>[s*1.589,y,z],(z,y)=>[s*1.548,y,z],'silver',s===-1);
    gm.ring(h,h,(z,y)=>[s*1.589,y,z],(z,y)=>[s*1.559,y,z],'black',s===1);
    gg.plane(h,(z,y)=>[s*1.578,y,z],null,'glass',[s,0,0]);
    // Slim rubber meeting edge; this moves with the leaf.
    let closeZ=leaf==='A'?z0:z1;gm.box(s<0?-1.592:1.572,1.12,closeZ-.006,s<0?-1.572:1.592,3.015,closeZ+.006,'black');
    let local=[-(origin[2]-outer[2]),0,0];
    node(`Door_${label}${i+1}_Leaf_${leaf}`,{Body:g.doorLocal(origin),Metal:gm.doorLocal(origin),Glass:gg.doorLocal(origin)},ctrl,local,{closedPosition:local,slideSign:sign,slideAxis:'X',stroke:.77,outerEdgeOrigin:true});
   }
  }
 }

 for(const [z,label] of [[4,'Rear'],[19,'Front']]){
  let b=new Geo();b.box(-1.11,.37,z-1.55,1.11,.72,z+1.55,'black');
  for(const s of [-1,1]){
   b.box(s<0?-1.28:1.03,.30,z-1.5,s<0?-1.03:1.28,.62,z+1.5,'metal');
   for(const dz of [-1.05,1.05]){
    // Stepped wheel profile: flange, tyre, inset face and hub.
    let x=s*.78;let rings=[[x-s*.145,.37],[x-s*.105,.43],[x+s*.085,.43],[x+s*.12,.34],[x+s*.135,.12]];
    for(let k=0;k<rings.length-1;k++)for(let j=0;j<16;j++){
     let a=j/16*2*Math.PI,c=(j+1)/16*2*Math.PI;
     const wp=(rr,t)=>[rr[0],.43+Math.cos(t)*rr[1],z+dz+Math.sin(t)*rr[1]];
     let pp=[wp(rings[k],a),wp(rings[k],c),wp(rings[k+1],c),wp(rings[k+1],a)];if(s<0)pp.reverse();b.quad(...pp,null,k===1?'metal':'black');
    }
    b.cylinder(0,[x+s*.15,.43,z+dz],.115,.08,8,'silver');
    b.box(s<0?-1.3:1.12,.36,z+dz-.2,s<0?-1.12:1.3,.65,z+dz+.2,'metal');
   }
   // Third-rail collection shoe remains below the passenger floor.
   b.box(s<0?-1.50:1.27,.34,z-.30,s<0?-1.27:1.50,.40,z+.30,'silver');
   b.box(s<0?-1.46:1.16,.40,z-.12,s<0?-1.16:1.46,.56,z+.12,'black');
   for(const dz of [-.43,.43])b.cylinder(1,[s*.96,.65,z+dz],.17,.14,10,'rubber');
  }
  for(const dz of [-1.05,1.05])b.cylinder(0,[0,.43,z+dz],.085,1.78,10,'metal');
  node(`Bogie_${label}`,{Metal:b.translate([0,.43,z])},root,[0,.43,z],{yawAxis:'Y'});
 }
 for(const [z,num] of [[6.55,1],[16.45,2]]){
  let ac=new Geo(),am=new Geo();const bottom=roundRect(-1.12,z-1.55,1.12,z+1.55,.10,2),top=roundRect(-1.00,z-1.43,1.00,z+1.43,.10,2);
  ac.ring(bottom,top,(x,zz)=>[x,3.51,zz],(x,zz)=>[x,3.70,zz],'grey',true);
  ac.plane(top,(x,zz)=>[x,3.70,zz],(x,zz)=>acUV(x,zz,z),'white',[0,1,0]);
  am.box(-1.05,3.49,z-1.38,1.05,3.515,z+1.38,'black');
  node(`Roof_AC_${num}`,{Body:ac.translate([0,3.49,z]),Metal:am.translate([0,3.49,z])},root,[0,3.49,z]);
 }
 const total=metrics.reduce((s,m)=>s+m.triangles,0);
 const encoder=await draco3d.createEncoderModule(),decoder=await draco3d.createDecoderModule();
 const io=new NodeIO().registerExtensions([KHRDracoMeshCompression]).registerDependencies({'draco3d.encoder':encoder,'draco3d.decoder':decoder});
 // Keep authored nodes and UV islands intact. Draco only, no destructive join/prune.
 await doc.transform(draco({method:'edgebreaker',encodeSpeed:5,decodeSpeed:5,quantizePosition:16,quantizeNormal:10,quantizeTexcoord:14}));
 await io.write(path.join(OUT,`c151-${kind}.glb`),doc);
 let manifest={kind,triangles:total,dimensions:[3.2,3.7,23],origin:[0,0,0],materials:Object.values(mats).map(m=>m.getName()),nodes:metrics,texture:[1024,1024],doorCenters};
 await fs.writeFile(path.join(DIR,'reports',`build-${kind}.json`),JSON.stringify(manifest,null,2));
 console.log(`${kind}: ${total} triangles; ${(await fs.stat(path.join(OUT,`c151-${kind}.glb`))).size} bytes`);
}
await build(true);await build(false);
