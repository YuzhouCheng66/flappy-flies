// Whitelisted static artifact: no fixtures, credentials, Python, or runtime deps.
import fs from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {createHash} from 'node:crypto';
import {COURSE_IDS} from '../campaign-selection.mjs';
const root=fileURLToPath(new URL('../',import.meta.url)),out=path.join(root,'dist');
await fs.mkdir(out,{recursive:true});
const sources=['index.html','game.css','game.mjs','contracts.js','race.js','race-clock.mjs','client.mjs','campaign-client.mjs','campaign-progress.mjs','campaign-difficulty.mjs','campaign-selection.mjs','campaign.mjs','replay-codec.mjs','inference-worker.mjs','gpu-model.mjs','gpu-device.mjs','engine.mjs','graph-loader.mjs','graph-layout.mjs','compact-model.mjs','backend-selector.mjs','README.md','THIRD_PARTY_NOTICES.md','validation.json'];
const contents=await Promise.all(sources.map(name=>fs.readFile(path.join(root,name),'utf8')));
const version=createHash('sha256').update(contents.join('\n')).digest('hex').slice(0,12);
for(let i=0;i<sources.length;i++){
 let text=contents[i];
 if(sources[i]==='campaign-client.mjs')text=text.replace("'./local-results/campaign/'","'./campaign/'");
 if(/\.(mjs|js|html)$/.test(sources[i]))text=text.replace(/(['"])(\.{1,2}\/[^'"\s?]+\.(?:mjs|js|css|json|bin))\1/g,(_,quote,url)=>`${quote}${url}?v=${version}${quote}`);
 if(sources[i]==='index.html')text=text.replace('<head>',`<head><meta name="application-version" content="${version}">`);
 await fs.writeFile(path.join(out,sources[i]),text);
}
await fs.mkdir(path.join(out,'model'),{recursive:true});
for(const name of ['model.json','weights.bin'])await fs.copyFile(path.join(root,'model',name),path.join(out,'model',name));
await fs.cp(path.join(root,'graph'),path.join(out,'graph'),{recursive:true});
const campaignRoot=path.join(root,'local-results','campaign'),manifest=JSON.parse(await fs.readFile(path.join(campaignRoot,'manifest.json'),'utf8'));
if(!manifest.complete||manifest.accepted!==100)throw Error('Campaign release requires 100 accepted courses');
await fs.mkdir(path.join(out,'campaign'),{recursive:true});
const selected=COURSE_IDS.map(id=>manifest.levels.find(e=>e.id===id));
if(selected.some(e=>!e))throw Error('Selected course is missing');
await fs.writeFile(path.join(out,'campaign','manifest.json'),JSON.stringify({...manifest,sourceAccepted:manifest.accepted,accepted:selected.length,requested:selected.length,bytes:selected.reduce((n,e)=>n+e.bytes,0),displayOrder:COURSE_IDS,levels:selected},null,2));
for(const entry of selected){
 if(!/^level-\d{3}\.bin\.gz$/.test(entry.file))throw Error('Invalid campaign artifact filename');
 const packed=await fs.readFile(path.join(campaignRoot,entry.file));
 if(createHash('sha256').update(packed).digest('hex')!==entry.sha256)throw Error(`Campaign hash mismatch: ${entry.id}`);
 await fs.writeFile(path.join(out,'campaign',entry.file),packed);
}
const brainIndex=JSON.parse(await fs.readFile(path.join(campaignRoot,'brains.json'),'utf8'));
for(const entry of selected){
 const trace=brainIndex.levels[entry.id];
 if(!trace||trace.recordingSha256!==entry.sha256||trace.weightsSha256!==entry.weightsSha256||trace.agents.length!==7)throw Error(`Missing matched CNS recordings: ${entry.id}`);
 for(const agent of trace.agents){
  if(!/^brain-\d{3}-0[2-8]\.bin\.gz$/.test(agent.file))throw Error('Invalid CNS artifact filename');
  const bytes=await fs.readFile(path.join(campaignRoot,agent.file));
  if(createHash('sha256').update(bytes).digest('hex')!==agent.sha256)throw Error('CNS artifact hash mismatch');
  await fs.writeFile(path.join(out,'campaign',agent.file),bytes);
 }
}
await fs.writeFile(path.join(out,'campaign','brains.json'),JSON.stringify(brainIndex,null,2));
await fs.writeFile(path.join(out,'.nojekyll'),'');
console.log(`Static site ready in dist/ · ${version}`);
