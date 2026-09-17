// Whitelisted static artifact: no fixtures, credentials, Python, or runtime deps.
import fs from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
const root=fileURLToPath(new URL('../',import.meta.url)),out=path.join(root,'dist');
await fs.mkdir(out,{recursive:true});
for(const name of ['index.html','game.css','game.mjs','contracts.js','race.js','client.mjs','inference-worker.mjs','gpu-model.mjs','engine.mjs','graph-loader.mjs','README.md','THIRD_PARTY_NOTICES.md','validation.json'])await fs.copyFile(path.join(root,name),path.join(out,name));
await fs.mkdir(path.join(out,'model'),{recursive:true});
for(const name of ['model.json','weights.bin'])await fs.copyFile(path.join(root,'model',name),path.join(out,'model',name));
await fs.cp(path.join(root,'graph'),path.join(out,'graph'),{recursive:true});
await fs.writeFile(path.join(out,'.nojekyll'),'');
console.log('Static site ready in dist/');
