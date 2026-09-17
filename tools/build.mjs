// Whitelisted static artifact: no fixtures, credentials, Python, or runtime deps.
import fs from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {createHash} from 'node:crypto';
const root=fileURLToPath(new URL('../',import.meta.url)),out=path.join(root,'dist');
await fs.mkdir(out,{recursive:true});
const sources=['index.html','game.css','game.mjs','contracts.js','race.js','race-clock.mjs','client.mjs','inference-worker.mjs','gpu-model.mjs','gpu-device.mjs','engine.mjs','graph-loader.mjs','README.md','THIRD_PARTY_NOTICES.md','validation.json'];
const contents=await Promise.all(sources.map(name=>fs.readFile(path.join(root,name),'utf8')));
const version=createHash('sha256').update(contents.join('\n')).digest('hex').slice(0,12);
for(let i=0;i<sources.length;i++){
 let text=contents[i];
 if(/\.(mjs|js|html)$/.test(sources[i]))text=text.replace(/(['"])(\.{1,2}\/[^'"\s?]+\.(?:mjs|js|css|json|bin))\1/g,(_,quote,url)=>`${quote}${url}?v=${version}${quote}`);
 if(sources[i]==='index.html')text=text.replace('<head>',`<head><meta name="application-version" content="${version}">`);
 await fs.writeFile(path.join(out,sources[i]),text);
}
await fs.mkdir(path.join(out,'model'),{recursive:true});
for(const name of ['model.json','weights.bin'])await fs.copyFile(path.join(root,'model',name),path.join(out,'model',name));
await fs.cp(path.join(root,'graph'),path.join(out,'graph'),{recursive:true});
await fs.writeFile(path.join(out,'.nojekyll'),'');
console.log(`Static site ready in dist/ · ${version}`);
