import fs from 'node:fs';
import {gunzipSync} from 'node:zlib';
import {decodeReplay} from '../replay-codec.mjs';
const file=process.argv[2];if(!file)throw Error('Pass a completed 100-layout evaluation JSON');
const audit=JSON.parse(fs.readFileSync(file));if(audit.total!==100||new Set(audit.rows.map(r=>r.level)).size!==100)throw Error('Full 100-layout audit required');
const root=new URL('../local-results/campaign/',import.meta.url),pairs=[];
for(const current of audit.rows.filter(r=>r.level<=60)){
 const r=JSON.parse(fs.readFileSync(new URL(`level-${String(current.level).padStart(3,'0')}.json`,root))),{frames}=decodeReplay(new Uint8Array(gunzipSync(fs.readFileSync(new URL(r.file,root)))));
 let backwardDistance=0,squared=0;
 for(let i=1;i<frames.length;i++){backwardDistance+=Math.max(0,frames[i-1].state[0]-frames[i].state[0]);if(i>1)for(let a=0;a<8;a++)for(let k=0;k<2;k++)squared+=(frames[i].forces[a][k]-frames[i-1].forces[a][k])**2;}
 pairs.push({level:current.level,original:{success:r.success,ticks:r.ticks,pathLength:r.pathLength,rotationRad:r.rotationRad,contacts:r.contacts,backwardDistance,forceChangeRms:Math.sqrt(squared/(16*Math.max(1,r.ticks-1)))},current});
}
const keys=['ticks','pathLength','rotationRad','contacts','backwardDistance','forceChangeRms'],matched=pairs.filter(p=>p.original.success&&p.current.success),mean=(side,key)=>matched.reduce((s,p)=>s+p[side][key],0)/matched.length;
const result={weights:audit.model,totalSuccesses:audit.successes,total:100,oldSuccesses:pairs.filter(p=>p.current.success).length,oldTotal:60,matchedSuccessfulPairs:matched.length,means:Object.fromEntries(keys.map(k=>[k,{original:mean('original',k),current:mean('current',k),percentChange:100*(mean('current',k)/mean('original',k)-1)}])),failed:audit.rows.filter(r=>!r.success).map(r=>r.level),pairs};
fs.writeFileSync(file.replace('.eval.json','.behavior.json'),JSON.stringify(result,null,2));console.log(JSON.stringify({...result,pairs:undefined},null,2));
