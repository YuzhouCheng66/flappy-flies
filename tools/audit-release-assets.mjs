// Keep the static campaign's exact dependency set in Git. Local research files
// are retained when --untrack-unused is used; no filesystem deletion occurs.
import fs from 'node:fs';
import {execFileSync} from 'node:child_process';
import {COURSE_IDS} from '../campaign-selection.mjs';
const prefix='local-results/campaign/',root=new URL('../',import.meta.url),brains=JSON.parse(fs.readFileSync(new URL(prefix+'brains.json',root)));
const keep=new Set(['manifest.json','brains.json',...COURSE_IDS.map(id=>`level-${String(id).padStart(3,'0')}.bin.gz`),...COURSE_IDS.flatMap(id=>brains.levels[id].agents.map(a=>a.file))].map(p=>prefix+p));
const rows=execFileSync('git',['ls-tree','-rl','HEAD'],{cwd:root,encoding:'utf8'}).trim().split('\n').map(s=>s.match(/^\d+ blob \w+\s+(\d+)\t(.+)$/)).filter(Boolean).map(m=>({bytes:Number(m[1]),file:m[2]}));
const redundant=rows.filter(r=>r.file.startsWith(prefix)&&!keep.has(r.file));
for(const file of keep)if(!fs.existsSync(new URL(file,root)))throw Error(`Missing release asset: ${file}`);
console.log(JSON.stringify({trackedBytes:rows.reduce((n,r)=>n+r.bytes,0),requiredCampaignFiles:keep.size,redundantCampaignFiles:redundant.length,redundantBytes:redundant.reduce((n,r)=>n+r.bytes,0),localFilesPreserved:true},null,2));
if(process.argv.includes('--untrack-unused'))for(let i=0;i<redundant.length;i+=40)execFileSync('git',['rm','--cached','--',...redundant.slice(i,i+40).map(r=>r.file)],{cwd:root,stdio:'ignore'});
