import fs from 'node:fs';
import {createHash} from 'node:crypto';
import {gunzipSync} from 'node:zlib';
import {createRequire} from 'node:module';
import {decodeReplay} from '../replay-codec.mjs';
import {campaignMeta} from '../campaign.mjs';
import {Engine} from '../engine.mjs';
const root=new URL('../local-results/campaign/',import.meta.url),base=JSON.parse(fs.readFileSync(new URL('../model/model.json',import.meta.url))),require=createRequire(import.meta.url),R=require('../race.js'),C=require('../contracts.js');
const levels=[],pending=[];
for(let id=1;id<=100;id++){
 const name=`level-${String(id).padStart(3,'0')}`,path=new URL(name+'.json',root),playerPath=new URL(name+'.player.json',root);
 if(!fs.existsSync(path)||!fs.existsSync(playerPath)){pending.push(id);continue;}
 const record=JSON.parse(fs.readFileSync(path)),player=JSON.parse(fs.readFileSync(playerPath)),packed=fs.readFileSync(new URL(record.file,root));
 if(createHash('sha256').update(packed).digest('hex')!==record.sha256||player.sourceReplay!==record.sha256)throw Error('Stale/corrupt level '+id);
 const {header,frames}=decodeReplay(new Uint8Array(gunzipSync(packed))),m=campaignMeta(base,record.layout),d=new Engine(m,null).reset(record.seed);
 if(!record.success||!frames.at(-1).success||frames.at(-1).stage!==record.layout.wall_x.length)throw Error('Incomplete fly course '+id);
 for(let i=1;i<frames.length;i++)if(C.collides(frames[i].state,d)||!R.safeMotion(frames[i-1].state,frames[i].state,d))throw Error('Unsafe replay '+id);
 if(!player.success){pending.push(id);continue;}
 // Re-execute the saved digital key sequence, without the route follower.
 const race=R.create(d),keyNames=[...R.KEYS];
 for(const command of player.best.commands){const keys=new Set(keyNames.filter((_,j)=>command.mask&(1<<j)));for(let k=0;k<command.steps;k++)R.advance(race,d,keys,R.SETTINGS.step);}
 if(race.winner!=='you'||Math.abs(race.time-player.best.seconds)>1e-7)throw Error('Keyboard replay failed '+id);
 // Arcade pacing targets a known executable score, not an unproved optimum.
 const slack=1.80-.77*(id-1)/99,flyRaceSeconds=player.best.seconds*slack;
 levels.push({id,file:record.file,sha256:record.sha256,bytes:record.bytes,seed:record.seed,weightsSha256:header.weightsSha256,checkpointSha256:header.checkpointSha256,walls:record.layout.wall_x.length,narrowWalls:record.layout.narrow_count,apertures:record.layout.apertures,gapCenters:record.layout.gap_y,playerDockDegrees:record.layout.player_dock_degrees,ticks:record.ticks,contacts:record.contacts,minClearance:record.minClearance,sourceSimulationSeconds:header.sourceSimulationSeconds,playerLowerBoundSeconds:player.lowerBoundSeconds,executablePlayerSeconds:player.best.seconds,flyRaceSeconds,referenceSlack:slack,proposedReplayRate:header.sourceSimulationSeconds/flyRaceSeconds,wallSecondsAt3_3x:flyRaceSeconds/3.3,brainMaxDisplayError:record.maxBrainDisplayError});
}
const sources=Object.fromEntries(['campaign.mjs','engine.mjs','gpu-model.mjs','race.js','replay-codec.mjs','tools/generate-campaign.mjs','tools/benchmark-player.mjs'].map(p=>[p,createHash('sha256').update(fs.readFileSync(new URL('../'+p,import.meta.url))).digest('hex')]));
const manifest={schema:'flappy-campaign-v1',generatedAt:new Date().toISOString(),complete:levels.length===100,requested:100,accepted:levels.length,pending,bytes:levels.reduce((s,l)=>s+l.bytes,0),sourceModel:{weights:base.weights_sha256,checkpoint:base.checkpoint_sha256,neurons:base.neurons,edges:25563197,agents:8,recurrences:4},sources,semantics:{motion:'precomputed full-model trajectories; unchanged float32 poses',brain:'2048 actual displayed neuron states on EVERY control tick; signed-int16 display encoding',messages:'all 20 directed links and all 4 sweeps, recorded real message-derived strengths',player:'live unchanged keyboard physics, per-level docking angle',pacing:'proposed arcade replay time scaling against executable keyboard reference; not live inference, not model physics speed, not an optimality proof',selection:'curated successful scenarios; NOT an unbiased success-rate evaluation'},levels};
// A curated archive may contain both original and adapted full-model runs.
// Identity belongs to each recording, never infer it from the base config.
manifest.sourceModels=Array.from(new Map(levels.map(l=>[l.weightsSha256,{weights:l.weightsSha256,checkpoint:l.checkpointSha256}])).values());
manifest.sourceModel={neurons:base.neurons,edges:25563197,agents:8,recurrences:4,identity:'per-level weightsSha256 and checkpointSha256; see sourceModels'};
fs.writeFileSync(new URL('manifest.json',root),JSON.stringify(manifest,null,2));
console.log(JSON.stringify({accepted:manifest.accepted,pending,MB:manifest.bytes/1e6,complete:manifest.complete,allAcceptedHaveExecutableKeyboardReference:true}));
