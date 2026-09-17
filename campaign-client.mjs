// Curated full-model recordings; player physics remains live on this computer.
// No neural weights, connectome download, WebGPU adapter or inference worker.
import {decodeReplay} from './replay-codec.mjs';
import {Engine} from './engine.mjs';
import {campaignMeta} from './campaign.mjs';
import {campaignDifficulty} from './campaign-difficulty.mjs';
import {sourceCourse,COURSE_COUNT,previewLevel} from './campaign-selection.mjs';

const root=new URL('./local-results/campaign/',import.meta.url);
let catalogue,meta,generation=0;
const cache=new Map();
let brainIndex;
export async function loadRecordedBrain(description,agent){
 if(!Number.isInteger(agent)||agent<1||agent>7)throw Error('Invalid additional CNS index');
 brainIndex??=json(new URL('brains.json',root));
 const index=await brainIndex,level=index.levels[description.campaign.sourceLevel];
 if(!level||level.weightsSha256!==description.student_sha256)throw Error('CNS recording unavailable for this checkpoint');
 const entry=level.agents.find(a=>a.agent===agent+1),response=await fetch(new URL(entry.file,root));
 if(!response.ok)throw Error('CNS recording unavailable');
 const packed=await response.arrayBuffer(),digest=Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',packed)),x=>x.toString(16).padStart(2,'0')).join('');
 if(digest!==entry.sha256)throw Error('CNS recording integrity mismatch');
 const bytes=await new Response(new Blob([packed]).stream().pipeThrough(new DecompressionStream('gzip'))).arrayBuffer();
 if(bytes.byteLength!==level.frames*2048*2)throw Error('CNS recording length mismatch');
 const view=new DataView(bytes),values=new Int16Array(bytes.byteLength/2);
 for(let i=0;i<values.length;i++)values[i]=view.getInt16(i*2,true);
 return values;
}
async function json(url){const r=await fetch(url);if(!r.ok)throw Error(`Level data unavailable (${r.status})`);return r.json();}
export function prepareRecording(record,entry,base,level=entry.id,total=100){
 const {header:h,frames}=record;
 if(h.level!==entry.id||h.weightsSha256!==entry.weightsSha256||!frames.at(-1)?.success)throw Error('Unverified campaign recording');
 const description=new Engine(campaignMeta(base,h.layout),null).reset(h.layout.seed);
 Object.assign(description,campaignDifficulty(level,total));
 // Presentation retiming only: recorded positions, actions and neuron states
 // are unchanged. The player still uses the original fixed-step physics.
 description.source_dt=h.dt;description.dt=h.dt/(entry.proposedReplayRate*description.fly_speed_factor);
 description.snapshot=frames[0];description.state=frames[0].state.slice();
 description.student_sha256=h.weightsSha256;
 description.campaign={level,total,sourceLevel:entry.id,bytes:entry.bytes,rate:entry.proposedReplayRate};
 return{description,frames};
}
async function load(level){
 const entry=catalogue.levels.find(x=>x.id===sourceCourse(level));
 if(!entry)throw Error('Selected course recording is unavailable');
 if(!cache.has(level)){
  const response=await fetch(new URL(entry.file,root));if(!response.ok)throw Error(`Level ${level} unavailable`);
  const packed=await response.arrayBuffer();
  const digest=Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',packed)),x=>x.toString(16).padStart(2,'0')).join('');
  if(digest!==entry.sha256)throw Error('Level integrity check failed');
  const stream=new Blob([packed]).stream().pipeThrough(new DecompressionStream('gzip'));
  const record=decodeReplay(new Uint8Array(await new Response(stream).arrayBuffer()));
  cache.set(level,record);if(cache.size>3)cache.delete(cache.keys().next().value);
 }
 return{...prepareRecording(cache.get(level),entry,meta,level,COURSE_COUNT),generation:++generation};
}
export async function campaignAPI(path,body={}){
 if(!catalogue)[catalogue,meta]=await Promise.all([json(new URL('manifest.json',root)),json(new URL('./model/model.json',import.meta.url))]);
 const level=path.split('/').pop()==='init'?(previewLevel(location.search,location.hostname)??Number(body.seed??1)):Number(body.seed);
 return load(level);
}
