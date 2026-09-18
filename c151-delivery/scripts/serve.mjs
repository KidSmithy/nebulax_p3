import http from 'node:http';
import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const types={'.html':'text/html','.js':'text/javascript','.mjs':'text/javascript','.css':'text/css','.png':'image/png','.glb':'model/gltf-binary','.wasm':'application/wasm','.json':'application/json'};
http.createServer(async(req,res)=>{
 try{
  if(req.method==='POST' && req.url==='/__capture'){
   if(req.headers.origin!==`http://${req.headers.host}`)throw new Error('origin');
   let chunks=[],size=0;for await(const c of req){size+=c.length;if(size>8*1024*1024)throw new Error('size');chunks.push(c);}
   const {name,png}=JSON.parse(Buffer.concat(chunks).toString());
   if(!/^c151-(cab|intermediate|consist)-(perspective|front|side|roof)(-doors-open)?\.png$/.test(name)||!png.startsWith('data:image/png;base64,'))throw new Error('capture');
   const dir=path.join(root,'reports/renders');await fs.mkdir(dir,{recursive:true});
   await fs.writeFile(path.join(dir,name),Buffer.from(png.split(',')[1],'base64'));
   res.writeHead(200,{'Content-Type':'application/json'});res.end(JSON.stringify({path:'/reports/renders/'+name}));return;
  }
  let url=decodeURIComponent(new URL(req.url,'http://localhost').pathname);
  if(url.endsWith('/'))url+='index.html';
  const file=path.resolve(root,'.'+url);
  if(!file.startsWith(root+path.sep))throw new Error('outside root');
  const data=await fs.readFile(file);res.writeHead(200,{'Content-Type':types[path.extname(file)]||'application/octet-stream','Cache-Control':'no-store'});res.end(data);
 }catch{res.writeHead(404);res.end('Not found');}
}).listen(43151,'127.0.0.1',()=>console.log('C151 preview: http://127.0.0.1:43151/preview/'));
