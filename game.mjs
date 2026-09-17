'use strict'; // Browser-native module; no remote inference endpoint.
import {api,client} from './client.mjs';
import {RACE_SPEED,LOOKAHEAD_TICKS,canStream,raceDuration,TICK_BUDGET_MS} from './race-clock.mjs';
const canvas=document.getElementById('world'),mainContext=canvas.getContext('2d'),C=CargoContracts,R=CargoRace;
let ctx=mainContext;
const brainSurface=document.createElement('canvas'),brainContext=brainSurface.getContext('2d');
let brainPaintAt=-Infinity,brainPaintKey='';
const ui=Object.fromEntries(['run','reset','random','seed','status','passed','total'].map(k=>[k,document.getElementById(k)]));
let W=innerWidth,H=innerHeight,description=null,generation=0,epoch=0;
let frames=[],clock=0,running=false,busy=false,resetting=false,ended=false,lastTime=0,lastPaintTime=0,camera=-.5;
let brainXY=[],brainEdges=[],brainOrder=[],latest=null,packetEvents=[],lastPacketTick=0,renderFault=false,terminalShown=false;
let recorder=null,captureParts=[],captureIdentity=null,captureOutcome=null,captureFinishAt=null;
let race=null,playerMain=false,playerCamera=-.5;
let raceReady=false,computeMode='warming',samples=[],msPerTick=0,gpuInfo=null,rebuffering=false,playAudit=null;
const heldKeys=new Set(),victory=document.getElementById('victory'),switchButton=document.getElementById('switch');
const recordButton=document.getElementById('record');
// Reused code-native sprites avoid hundreds of expensive per-neuron shadow
// blur passes each frame; measured node positions/activation remain unchanged.
const neuronGlow=['#64ffc6','#ff7cd2'].map(color=>{
 const sprite=document.createElement('canvas');sprite.width=sprite.height=40;const g=sprite.getContext('2d'),gradient=g.createRadialGradient(20,20,0,20,20,20);
 gradient.addColorStop(0,color);gradient.addColorStop(.15,color+'b0');gradient.addColorStop(1,color+'00');g.fillStyle=gradient;g.fillRect(0,0,40,40);return sprite;
});
const speed=RACE_SPEED,bufferTarget=LOOKAHEAD_TICKS;
const gpuBadge=document.querySelector('.live'),gpuDetails=document.getElementById('gpu-details');
function showGPU(){
 const adapter=gpuInfo?.adapter||{},name=adapter.device||[adapter.vendor,adapter.architecture].filter(Boolean).join(' ')||'Detecting';
 gpuBadge.textContent=`WebGPU · ${adapter.vendor||'checking'}`;
 gpuDetails.textContent=`Device: ${name}\nBackend: WebGPU compute shaders${gpuInfo?.kernel?' · '+gpuInfo.kernel:''}\nHigh-performance GPU requested; browser/OS decides exposed adapters.\n${msPerTick?`Control: ${msPerTick.toFixed(1)} ms/tick · budget ${TICK_BUDGET_MS.toFixed(1)} ms\n`:''}Race: fixed ${speed.toFixed(2)}× · ${computeMode==='stream'?'live inference':computeMode==='prepared'?'local precomputation (not real-time inference)':'warming up'}\nNo server inference; no downloaded trajectories.`;
}
function makeRaceReady(){
 if(raceReady)return;raceReady=true;running=false;clock=0;race=R.create(description);playerCamera=camera;heldKeys.clear();
 ui.run.textContent='Run';ui.status.textContent=computeMode==='stream'?'Ready · full-speed race':'Locally prepared · Run for full-speed race';
}
function stageSplit(){return W*.60;}
function panels(){const split=stageSplit(),top=92,bottom=H-36,gap=12,right=split+6,w=W-right-14,bh=(bottom-top-gap)*.46;return{main:{x:14,y:top,w:split-26,h:bottom-top},brain:{x:right,y:top,w,h:bh},mini:{x:right,y:top+bh+gap,w,h:bottom-top-bh-gap}};}
function resize(){W=innerWidth;H=innerHeight;const dpr=Math.min(devicePixelRatio,2);canvas.width=W*dpr;canvas.height=H*dpr;ctx.setTransform(dpr,0,0,dpr,0,0);document.documentElement.style.setProperty('--stage-split',`${stageSplit()}px`);}
addEventListener('resize',resize);resize();
function stop(message){running=false;heldKeys.clear();ui.run.textContent=raceReady?'Run':'Practice';ui.status.textContent=message;if(recorder?.state==='recording')recorder.pause();}
function finishCapture(){if(recorder&&recorder.state!=='inactive'){captureOutcome={success:!!latest?.success,stage:latest?.stage||0,winner:race?.winner};recorder.stop();}}
function toggleCapture(){
 if(recorder){finishCapture();return;}if(!raceReady||!description||race?.winner||!window.MediaRecorder)return;
 const mime=['video/mp4;codecs=avc1.42001E','video/webm;codecs=vp9','video/webm'].find(m=>MediaRecorder.isTypeSupported(m));
 if(!mime){ui.status.textContent='Video recording is unavailable in this browser';return;}
 const stream=canvas.captureStream(30);captureParts=[];captureOutcome=null;captureFinishAt=null;
 captureIdentity={seed:description.layout.seed,checkpoint:(description.student_sha256||'unknown').slice(0,12),fullCheckpoint:description.student_sha256,gates:description.layout.wall_x.length,epoch};
 try{recorder=new MediaRecorder(stream,{mimeType:mime,videoBitsPerSecond:6000000});}
 catch(e){stream.getTracks().forEach(t=>t.stop());ui.status.textContent=e.message;return;}
 recorder.ondataavailable=e=>{if(e.data.size)captureParts.push(e.data);};
 recorder.onstop=async()=>{
  const result=captureOutcome?.winner?`${captureOutcome.winner}-win`:captureOutcome?.success?'clear':`partial-${captureOutcome?.stage||0}-of-${captureIdentity.gates}`;
  const blob=new Blob(captureParts,{type:mime}),url=URL.createObjectURL(blob),link=document.createElement('a');
  link.href=url;link.download=`flappy-flies-${captureIdentity.seed}-${captureIdentity.checkpoint}-${result}.${mime.startsWith('video/mp4')?'mp4':'webm'}`;
  stream.getTracks().forEach(t=>t.stop());
  link.click();ui.status.textContent='Clip saved on your device';
  setTimeout(()=>URL.revokeObjectURL(url),10000);
  recorder=null;captureParts=[];document.body.classList.remove('capturing');recordButton.classList.remove('recording');recordButton.setAttribute('aria-label','Record live run');
 };
 recorder.start(1000);document.body.classList.add('capturing');recordButton.classList.add('recording');recordButton.setAttribute('aria-label','Stop recording');
}
recordButton.onclick=toggleCapture;
function setDescription(d){
 raceReady=false;computeMode='warming';samples=[];msPerTick=0;rebuffering=false;playAudit=null;
 description=d;camera=d.state[0]+.5;clock=d.snapshot.tick;packetEvents=[];lastPacketTick=0;frames=[d.snapshot];latest=frames[0];ended=d.snapshot.done;renderFault=false;terminalShown=false;captureFinishAt=null;
 race=R.create(d);playerCamera=camera;heldKeys.clear();victory.hidden=true;document.getElementById('again').disabled=false;
 ui.total.textContent=String(d.layout.wall_x.length).padStart(2,'0');ui.passed.textContent=String(d.snapshot.stage).padStart(2,'0');
 // Orthographic camera rotation of measured 3-D coordinates: oblique view,
 // not a deformation of anatomy to match a decorative silhouette.
 const xyz=d.brain_coordinates_3d||[],xy=xyz.length?xyz.map(p=>{
  const yaw=.56,tilt=.94,roll=-.20,c=Math.cos(yaw),s=Math.sin(yaw),ct=Math.cos(tilt),st=Math.sin(tilt);
  const x=c*p[0]-s*p[2],z=s*p[0]+c*p[2],y=ct*p[1]+st*z;
  return[Math.cos(roll)*x-Math.sin(roll)*y,Math.sin(roll)*x+Math.cos(roll)*y,-st*p[1]+ct*z];
 }):(d.brain_coordinates||[]).map(p=>[...p,0]);brainXY=[];
 if(xy.length){let xmin=Infinity,xmax=-Infinity,ymin=Infinity,ymax=-Infinity,zmin=Infinity,zmax=-Infinity;for(const p of xy){xmin=Math.min(xmin,p[0]);xmax=Math.max(xmax,p[0]);ymin=Math.min(ymin,p[1]);ymax=Math.max(ymax,p[1]);zmin=Math.min(zmin,p[2]);zmax=Math.max(zmax,p[2]);}const extent=Math.max(xmax-xmin,ymax-ymin);brainXY=xy.map(p=>[(p[0]-(xmin+xmax)/2)/extent,(p[1]-(ymin+ymax)/2)/extent,(p[2]-zmin)/(zmax-zmin||1)]);}
 brainEdges=d.brain_edges||[];brainOrder=brainXY.map((_,i)=>i).sort((i,j)=>brainXY[i][2]-brainXY[j][2]);
 ui.run.disabled=false;ui.run.textContent='Practice';ui.status.textContent=client.ready?'Warming up your GPU · Practice now':'Practice while the full fly brain loads';
}
async function reset(seed=Number(ui.seed.value)){
 if(resetting)return;finishCapture();resetting=true;epoch++;running=false;heldKeys.clear();ui.run.disabled=true;ui.run.textContent='Run';ui.status.textContent='Resetting…';
 try{const data=await api('/api/reset',{seed});generation=data.generation;setDescription(data.description);}
 catch(e){ui.status.textContent=e.message;}finally{resetting=false;}
}
async function prefetch(){
 if(!client.ready||busy||resetting||document.hidden||race?.winner||ended||!description)return;
 if(computeMode==='stream'&&!rebuffering&&frames.at(-1).tick-clock>=bufferTarget)return;
 busy=true;const current=generation,requestEpoch=epoch,after=frames.at(-1).tick;
 try{
  const data=await api('/api/step',{generation:current,after_tick:after,count:8});
  if(requestEpoch!==epoch||race?.winner)return;
  if(data.stale||data.generation!==current)throw Error('Session changed · Reset');
  msPerTick=msPerTick ? .8*msPerTick+.2*data.msPerTick : data.msPerTick;
  if(computeMode==='warming'){
   samples.push(data.msPerTick);
   if(samples.length>=4){const sorted=samples.slice(1).sort((a,b)=>a-b);computeMode=canStream(sorted.at(-1))?'stream':'prepared';}
  }
  showGPU();
  C.validateBatch(after,data.frames,description);frames.push(...data.frames);
  for(const f of data.frames){
   if(!f.natural)continue;
   const energy=f.natural.map((sweep,s)=>sweep.map((vec,j)=>Math.sqrt(vec.reduce((v,h,k)=>v+h*h/Math.max(f.precision[s][j][k],1e-8),0)/3)));
   const scale=Math.max(1e-8,...energy.flat());
   f.packets=[];for(let s=0;s<energy.length;s++)f.sender.forEach((u,j)=>f.packets.push({at:f.tick-1+s/energy.length,u,v:f.receiver[j],strength:Math.sqrt(energy[s][j]/scale)}));
  }ended=data.done;
  if(!raceReady){
   if(ended||(computeMode==='stream'&&data.frames.at(-1).tick>=64))makeRaceReady();
   else if(computeMode==='prepared')ui.status.textContent=`Preparing locally · gate ${data.frames.at(-1)?.stage||0}/4 · Practice now`;
  }
  if(rebuffering&&ended){rebuffering=false;ui.run.disabled=false;ui.run.textContent='Run';ui.status.textContent='Buffer ready · resume at full speed';}
 }catch(e){if(requestEpoch===epoch)stop(e.message);}finally{busy=false;}
}
ui.run.onclick=()=>{if(!description||resetting||renderFault||rebuffering)return;if(race?.winner){reset();return;}running=!running;ui.run.textContent=running?'Pause':raceReady?'Run':'Practice';if(raceReady)ui.status.textContent='';if(running){lastTime=performance.now();if(raceReady)playAudit={wall:lastTime,sim:race.time};if(recorder?.state==='paused')recorder.resume();prefetch();}else{heldKeys.clear();if(recorder?.state==='recording')recorder.pause();}};
ui.reset.onclick=()=>reset();ui.random.onclick=()=>{ui.seed.value=Math.floor(Math.random()*1000000);reset();};
ui.seed.addEventListener('change',()=>reset());
switchButton.onclick=()=>{playerMain=!playerMain;switchButton.setAttribute('aria-pressed',String(playerMain));};
document.getElementById('again').onclick=()=>{document.getElementById('again').disabled=true;reset();};
addEventListener('keydown',e=>{
 if(['INPUT','SELECT','TEXTAREA'].includes(document.activeElement.tagName))return;
 if(R.KEYS.has(e.code)){e.preventDefault();if(running&&!race?.winner)heldKeys.add(e.code);}
 if(e.code==='Space'&&!e.repeat&&!race?.winner){e.preventDefault();ui.run.click();}
});
addEventListener('keyup',e=>heldKeys.delete(e.code));
addEventListener('blur',()=>{heldKeys.clear();if(running)stop('Paused');});
document.addEventListener('visibilitychange',()=>{if(document.hidden&&running)stop('Paused');});
addEventListener('keydown',e=>{if(e.code==='KeyR'&&!['INPUT','SELECT'].includes(document.activeElement.tagName))toggleCapture();});
function path(points){ctx.beginPath();points.forEach((p,i)=>i?ctx.lineTo(...p):ctx.moveTo(...p));ctx.closePath();}
function line(a,b,color,width=1){ctx.strokeStyle=color;ctx.lineWidth=width;ctx.beginPath();ctx.moveTo(...a);ctx.lineTo(...b);ctx.stroke();}
function rounded(x,y,w,h,r=12){ctx.beginPath();ctx.roundRect(x,y,w,h,r);}
function noise(x){return Math.sin(x*127.1+31.7)*43758.5453%1;}

// Decorative scenery never participates in the physical collision scene.
function backdrop(width=W,height=H,viewCamera=camera){
 const W=width,H=height,camera=viewCamera;
 const sky=ctx.createLinearGradient(0,0,0,H);sky.addColorStop(0,'#21b7ff');sky.addColorStop(.48,'#68e0ff');sky.addColorStop(1,'#c8fff0');ctx.fillStyle=sky;ctx.fillRect(0,0,W,H);
 // Soft sculpted cloud banks; two parallax depths, no grey haze.
 for(let layer=0;layer<2;layer++)for(let i=-1;i<6;i++){
  const stride=280+layer*110,x=i*stride-((camera*(12+layer*6))%stride),y=H*(.27+layer*.23)+(i%3)*36,s=layer?.85:1.1;
  ctx.save();ctx.translate(x,y);ctx.scale(s,s);
  ctx.fillStyle=layer?'#dffff7':'#b2f5ff';rounded(-48,1,180,25,12);ctx.fill();
  ctx.fillStyle=layer?'#f4fff8':'#ecfcff';ctx.beginPath();ctx.arc(-18,0,24,Math.PI,0);ctx.arc(25,-15,37,Math.PI,0);ctx.arc(65,-2,25,Math.PI,0);ctx.arc(96,5,17,Math.PI,0);ctx.lineTo(113,13);ctx.lineTo(-42,13);ctx.closePath();ctx.fill();ctx.restore();
 }
 // Rounded tropical mesas with explicit lit / shaded facets.
 for(let layer=0;layer<3;layer++){
  const unit=220+layer*75,shift=camera*(12+layer*16),base=H*(.83+layer*.085);
  for(let i=-3;i<W/unit+4;i++){
   const x=i*unit-((shift%unit)+unit)%unit,peak=45+Math.abs(noise(i+layer*20))*95;
   ctx.fillStyle=['#69dfe5','#20cbbf','#04acaa'][layer];
   ctx.beginPath();ctx.moveTo(x-40,base+100);ctx.bezierCurveTo(x,base-peak*.4,x+unit*.15,base-peak,x+unit*.32,base-peak);ctx.bezierCurveTo(x+unit*.55,base-peak,x+unit*.62,base+35,x+unit+30,base+100);ctx.closePath();ctx.fill();
   ctx.fillStyle=['#a1f8d9','#63efbe','#33d8ae'][layer];path([[x+unit*.32,base-peak],[x+unit*.42,base-peak+15],[x+unit*.52,base+100],[x+unit*.12,base+100]]);ctx.fill();
  }
 }
}
function drawSolids(walls,payload,scale){
 const depth=C.extrusionDepth(scale),layers=Math.ceil(Math.hypot(...depth)*2);
 // Depth-slice painter: ALL solids at z=k precede ALL solids at z=k-1.
 // Thus a rear cargo slice cannot paint over a nearer wall. The old scheme
 // drew the entire cargo after every wall, incorrectly hiding wall surfaces.
 for(let k=layers;k>=1;k--){const u=k/layers;
  for(const points of walls){path(points.map(p=>[p[0]+depth[0]*u,p[1]+depth[1]*u]));ctx.fillStyle='#5730a0';ctx.fill();}
  for(const points of payload){path(points.map(p=>[p[0]+depth[0]*u,p[1]+depth[1]*u]));ctx.fillStyle='#df8416';ctx.fill();}
 }
 for(const points of walls){const g=ctx.createLinearGradient(points[0][0],0,points[1][0],0);g.addColorStop(0,'#c09aff');g.addColorStop(.22,'#a974ee');g.addColorStop(1,'#7338cc');path(points);ctx.fillStyle=g;ctx.fill();}
 const ys=payload.flat().map(p=>p[1]),g=ctx.createLinearGradient(0,Math.min(...ys),0,Math.max(...ys));g.addColorStop(0,'#fff291');g.addColorStop(.38,'#ffdb39');g.addColorStop(1,'#ffb41f');
 for(const points of payload){path(points);ctx.fillStyle=g;ctx.fill();}
}
function drawApple(screen,scale,time,width,success){
 const goal=description.layout.goals.at(-1),p=screen(goal),r=scale*.95;if(p[0]-r>width||p[0]+r<0)return;
 ctx.save();ctx.translate(...p);
 const shape=()=>{ctx.beginPath();ctx.moveTo(0,-r*.68);ctx.bezierCurveTo(-r*.70,-r*1.10,-r*1.07,-r*.38,-r*.92,r*.27);ctx.bezierCurveTo(-r*.75,r*1.03,-r*.18,r*1.17,0,r*.94);ctx.bezierCurveTo(r*.21,r*1.14,r*.78,r*.91,r*.91,r*.28);ctx.bezierCurveTo(r*1.10,-r*.44,r*.66,-r*1.09,0,-r*.68);ctx.closePath();};
 ctx.save();ctx.translate(7,9);shape();ctx.fillStyle='#90324b';ctx.fill();ctx.restore();
 const flesh=ctx.createRadialGradient(-r*.42,-r*.45,r*.03,r*.2,r*.35,r*1.5);flesh.addColorStop(0,'#ffbd79');flesh.addColorStop(.22,'#ff6965');flesh.addColorStop(.65,'#db3753');flesh.addColorStop(1,'#8b294c');shape();ctx.fillStyle=flesh;ctx.fill();
 ctx.save();shape();ctx.clip();
 for(const [x,y,s] of [[-.65,.22,.14],[.52,.55,.17],[.59,-.37,.12],[-.20,.78,.09]]){ctx.fillStyle='#843948';ctx.beginPath();ctx.ellipse(x*r,y*r,s*r,s*r*.68,.4,0,Math.PI*2);ctx.fill();ctx.strokeStyle='#ba6947';ctx.lineWidth=2;ctx.stroke();}
 ctx.strokeStyle='#ffd3a186';ctx.lineWidth=Math.max(3,r*.035);ctx.lineCap='round';ctx.beginPath();ctx.ellipse(-r*.49,-r*.39,r*.24,r*.34,.45,3.3,4.75);ctx.stroke();ctx.restore();
 ctx.strokeStyle='#683440';ctx.lineWidth=r*.085;ctx.beginPath();ctx.moveTo(0,-r*.72);ctx.quadraticCurveTo(-r*.1,-r*.91,r*.02,-r*1.12);ctx.stroke();
 ctx.fillStyle='#84cb45';ctx.beginPath();ctx.moveTo(r*.01,-r*.88);ctx.bezierCurveTo(r*.03,-r*1.20,r*.42,-r*1.24,r*.48,-r*1.06);ctx.bezierCurveTo(r*.36,-r*.84,r*.14,-r*.83,r*.01,-r*.88);ctx.fill();line([r*.02,-r*.90],[r*.39,-r*1.07],'#d8ed7c',1.5);
 // Stink curls are decorative only; the apple is a backdrop target, not an
 // extra physical obstacle. The inset socket is the EXACT terminal T pose.
 for(let i=0;i<3;i++){const phase=(time*.00017+i*.31)%1,x=(i-1)*r*.37,y=-r*.86-phase*r*.5;ctx.globalAlpha=(1-phase)*.45;ctx.strokeStyle='#9bce3c';ctx.lineWidth=3+i%2;ctx.beginPath();ctx.moveTo(x,y);ctx.bezierCurveTo(x-r*.17,y-r*.14,x+r*.17,y-r*.25,x,y-r*.41);ctx.stroke();}ctx.globalAlpha=1;
 ctx.restore();
 const slot=C.tOutline(description.rectangles).map(v=>screen(C.transform(v,goal)));
 // Edge highlights plus inward shadow read as a recess; never move or snap
 // the payload when docking. Success remains the server's physical criterion.
 path(slot);ctx.strokeStyle='#ffd4ab';ctx.lineWidth=7;ctx.lineJoin='round';ctx.stroke();
 path(slot);ctx.fillStyle='#682c3c';ctx.fill();ctx.save();path(slot);ctx.clip();path(slot.map(p=>[p[0]+4,p[1]+4]));ctx.fillStyle='#aa5651';ctx.fill();ctx.restore();
 if(success){path(slot);ctx.strokeStyle='#eeff9e';ctx.lineWidth=2;ctx.shadowColor='#edff78';ctx.shadowBlur=12;ctx.stroke();ctx.shadowBlur=0;}
}
function drawFly(x,y,angle,active){
 ctx.save();ctx.translate(x,y);ctx.rotate(angle);const z=Math.max(.8,Math.min(1.45,W/1150));ctx.scale(z,z);
 ctx.fillStyle='#09172466';ctx.beginPath();ctx.ellipse(3,6,7,5,0,0,Math.PI*2);ctx.fill();
 ctx.fillStyle='#e4f4edcc';ctx.strokeStyle='#fff8';ctx.lineWidth=.6;const flap=running?Math.sin(clock*2.5+x)*.17:0;
 ctx.beginPath();ctx.ellipse(-5,-1,3.5,7.8,-.5+flap,0,Math.PI*2);ctx.ellipse(5,-1,3.5,7.8,.5-flap,0,Math.PI*2);ctx.fill();ctx.stroke();
 for(let i=-1;i<=1;i++){line([-2,i*3],[-7,i*4+3],'#152333',1.1);line([2,i*3],[7,i*4+3],'#152333',1.1);}
 ctx.fillStyle='#172632';ctx.beginPath();ctx.ellipse(0,2,3.3,5.8,0,0,Math.PI*2);ctx.fill();ctx.fillStyle='#d9b877';ctx.fillRect(-2,2,4,1.6);
 ctx.fillStyle='#132334';ctx.beginPath();ctx.arc(0,-4.5,3,0,Math.PI*2);ctx.fill();
 if(active){ctx.strokeStyle='#b9ffeaa0';ctx.lineWidth=.8;ctx.beginPath();ctx.arc(0,0,14,0,Math.PI*2);ctx.stroke();}ctx.restore();
}
function brainPanel(box,a,b){
 const dpr=Math.min(devicePixelRatio,2),key=[box.x,box.y,box.w,box.h,dpr,!!a.brain].join(':'),now=performance.now();
 if(key!==brainPaintKey||(a.brain&&now-brainPaintAt>=1000/30)){
  const width=Math.ceil(box.w*dpr),height=Math.ceil(box.h*dpr);
  if(brainSurface.width!==width||brainSurface.height!==height){brainSurface.width=width;brainSurface.height=height;}
  brainContext.setTransform(1,0,0,1,0,0);brainContext.clearRect(0,0,width,height);brainContext.setTransform(dpr,0,0,dpr,-box.x*dpr,-box.y*dpr);
  ctx=brainContext;try{paintBrainPanel(box,a,b);}finally{ctx=mainContext;}
  brainPaintAt=now;brainPaintKey=key;
 }
 ctx.drawImage(brainSurface,box.x,box.y,box.w,box.h);
}
function paintBrainPanel(box,a,b){
 const {x,y,w,h}=box;if(w<90||h<100)return;
 const panel=ctx.createLinearGradient(x,y,x+w,y+h);panel.addColorStop(0,'#30236a');panel.addColorStop(1,'#171f51');ctx.fillStyle=panel;rounded(x,y,w,h,18);ctx.fill();ctx.strokeStyle='#c2acff99';ctx.lineWidth=1.5;ctx.stroke();line([x+22,y+63],[x+w-22,y+63],'#a695ff45');
 const previous=a.brain||[],mix=Math.min(1,Math.max(0,clock-a.tick));
 const rates=previous.map((v,i)=>v+((b.brain?.[i]??v)-v)*mix),scale=Math.min(w*.95,(h-82)*1.18),bx=x+w*.50,by=y+55+(h-91)*.46,max=Math.max(.03,...rates.map(Math.abs));
 ctx.save();rounded(x+5,y+55,w-10,Math.max(1,h-91),12);ctx.clip();
 // Measured pair connectivity, NOT invented morphology. Batch paths by
 // activity to keep 4,804 true graph links inexpensive at display frame rate.
 const palettes=['#6dc8ff','#ae87ff','#ff89ce','#91f7d3'];
 for(let group=0;group<8;group++){ctx.beginPath();let count=0;for(const [u,v,weight] of brainEdges){const amp=Math.min(1,(Math.abs(rates[u]||0)+Math.abs(rates[v]||0))/(2*max)),bucket=Math.min(7,Math.floor(amp*8));if(bucket!==group)continue;const p=brainXY[u],q=brainXY[v];if(!p||!q)continue;ctx.moveTo(bx+p[0]*scale,by+p[1]*scale);ctx.lineTo(bx+q[0]*scale,by+q[1]*scale);count++;}if(count){ctx.strokeStyle=palettes[Math.floor(group/2)];ctx.globalAlpha=.055+group*.023;ctx.lineWidth=.38+group*.025;ctx.stroke();}}ctx.globalAlpha=1;
 for(const i of brainOrder){
  const [px,py,depth]=brainXY[i],value=rates[i]||0,amp=Math.min(1,Math.abs(value)/max),change=Math.min(1,Math.abs(value-(previous[i]||0))*40),nx=bx+px*scale,ny=by+py*scale;
  ctx.fillStyle=rates.length?(value>=0?`rgba(125,244,226,${.35+amp*.65})`:`rgba(255,154,215,${.35+amp*.65})`):'#b4bcdfaa';
  if(change>.06){const glow=2+change*3;ctx.globalAlpha=.15+change*.25;ctx.drawImage(neuronGlow[value>=0?0:1],nx-glow,ny-glow,glow*2,glow*2);ctx.globalAlpha=1;}
  ctx.beginPath();ctx.arc(nx,ny,.35+depth*.3+amp*.65,0,Math.PI*2);ctx.fill();
 }
 ctx.restore();
 const rms=a.brain_rms||[],barW=(w-48)/8;
 for(let i=0;i<8;i++){
  const bx=x+24+i*barW,base=y+h-20;ctx.fillStyle='#1e384a';rounded(bx,base-14,barW-6,14,3);ctx.fill();
  const v=Math.min(14,(rms[i]||0)*240);ctx.fillStyle='#77d8bd';rounded(bx,base-v,barW-6,v,Math.min(3,v/2));ctx.fill();ctx.fillStyle='#81a9ae';ctx.font='8px "Segoe UI"';ctx.textAlign='center';ctx.fillText(String(i+1).padStart(2,'0'),bx+(barW-6)/2,base+12);
 }ctx.textAlign='left';
}
function drawWorld(box,q,viewCamera,isFlies,t){
 const split=box.w,H=box.h,camera=viewCamera;
 ctx.save();rounded(box.x,box.y,box.w,box.h,18);ctx.clip();ctx.translate(box.x,box.y);backdrop(split,H,camera);
 const scale=Math.min((H-62)/3.2,split/4.25),cy=H*.52,screen=p=>[split*.40+(p[0]-camera)*scale,cy-p[1]*scale];
 const floor=screen([0,-description.world_y])[1],ceiling=screen([0,description.world_y])[1];
 ctx.save();ctx.beginPath();ctx.rect(0,32,split,H-54);ctx.clip();
 // Track rails sit outside the physical world bounds.
 for(const [y,dir] of [[floor,1],[ceiling,-1]]){
  const g=ctx.createLinearGradient(0,y,0,y+dir*25);g.addColorStop(0,'#a98bff');g.addColorStop(1,'#653acd');ctx.fillStyle=g;ctx.fillRect(0,Math.min(y,y+dir*25),split,25);line([0,y],[split,y],'#efe0ff',2);
  for(let x=-((camera*scale)%70);x<split;x+=70)line([x,y+dir*5],[x-14,y+dir*21],'#d9c3ff70',3);
 }
 const course=description.layout;
 drawApple(screen,scale,t,split,isFlies?latest?.success:race.winner==='you');
 const wallPoints=description.colliders.map(poly=>poly.map(screen));
 const payload=description.rectangles.map(rect=>rect.map(p=>screen(C.transform(p,q))));
 drawSolids(wallPoints,payload,scale);
 description.colliders.forEach((poly,index)=>{
  const points=poly.map(screen),left=points[0][0],right=points[1][0],bottom=points[0][1],top=points[2][1];if(right< -30||left>split+30)return;
  const edge=index%2===0?top:bottom,rimY=index%2===0?edge+1:edge-5;
  // Rims and fasteners stay inside the actual collider face.
  ctx.fillStyle='#ffe76a';ctx.fillRect(left+1,rimY,right-left-2,4);ctx.shadowColor='#ebf876';ctx.shadowBlur=6;ctx.fillStyle='#fcffce';ctx.fillRect(left+2,index%2===0?edge+7:edge-9,right-left-4,1.4);ctx.shadowBlur=0;
  ctx.save();path(points);ctx.clip();for(let y=top+22;y<bottom-12;y+=48){line([left+4,y],[right-4,y],'#dfbaff55');ctx.fillStyle='#f4dcffcc';ctx.fillRect(left+4,y-5,2,2);ctx.fillRect(right-6,y-5,2,2);}line([left+4,top],[left+4,bottom],'#f2cdff70');ctx.restore();
  if(index%2===1){ctx.fillStyle='#c0d7d3';ctx.font='600 10px "Segoe UI"';ctx.textAlign='center';ctx.fillText(String(Math.floor(index/2)+1).padStart(2,'0'),(left+right)/2,Math.max(ceiling+18,bottom-23));ctx.textAlign='left';}
 });
 const handles=description.handles.map(p=>screen(C.transform(p,q)));
 for(const event of isFlies?packetEvents:[]){
  const age=clock-event.at;if(age<0||event.strength<.0001)continue;
  const phase=age/3,p0=handles[event.u],p1=handles[event.v],d=[p1[0]-p0[0],p1[1]-p0[1]],len=Math.hypot(...d)||1,n=[-d[1]/len,d[0]/len],unit=[d[0]/len,d[1]/len];
  const head=[p0[0]+d[0]*phase+n[0]*2,p0[1]+d[1]*phase+n[1]*2],tail=[p0[0]+d[0]*Math.max(0,phase-.36)+n[0]*2,p0[1]+d[1]*Math.max(0,phase-.36)+n[1]*2];
  ctx.shadowColor='#00ff88';ctx.shadowBlur=8;line(tail,head,`rgba(0,225,105,${.12+event.strength*.55})`,1.8);path([head,[head[0]-unit[0]*5+n[0]*2.5,head[1]-unit[1]*5+n[1]*2.5],[head[0]-unit[0]*5-n[0]*2.5,head[1]-unit[1]*5-n[1]*2.5]]);ctx.fillStyle='#30ff97dd';ctx.fill();ctx.shadowBlur=0;ctx.strokeStyle='#008e5e90';ctx.lineWidth=.65;ctx.stroke();
 }
 if(isFlies)handles.forEach((p,i)=>drawFly(p[0],p[1],-q[2]+Math.atan2(description.handles[i][1],description.handles[i][0])+Math.PI/2,i===0));ctx.restore();
 // Fixed stage indicators, not a scrubber: all state comes from physics.
 const mapX=20,mapY=H-15,mapW=Math.min(220,split-40),count=course.wall_x.length;line([mapX,mapY],[mapX+mapW,mapY],'#397b9690',2);
 for(let i=0;i<count;i++){const x=mapX+i*mapW/(count-1||1);ctx.fillStyle=i<(isFlies?latest.stage:race.stage)?'#ffffff':'#7852d1';ctx.beginPath();ctx.arc(x,mapY,4,0,Math.PI*2);ctx.fill();ctx.strokeStyle='#ffffffe0';ctx.lineWidth=1;ctx.stroke();}
 const progress=Math.max(0,Math.min(1,(q[0]-course.initial[0])/(course.goals.at(-1)[0]-course.initial[0])));ctx.fillStyle='#f8c08b';ctx.beginPath();ctx.arc(mapX+progress*mapW,mapY,3,0,Math.PI*2);ctx.fill();
 ctx.fillStyle='#effaffdb';rounded(10,8,82,23,8);ctx.fill();ctx.fillStyle='#443083';ctx.font='800 11px "Segoe UI"';ctx.fillText(isFlies?'● FLIES':'● YOU',21,24);
 ctx.restore();ctx.strokeStyle=isFlies?'#eaffffd0':'#fff38fcc';ctx.lineWidth=2;rounded(box.x,box.y,box.w,box.h,18);ctx.stroke();
}
function announceWinner(t){
 if(!race?.winner||terminalShown)return;
 terminalShown=true;running=false;heldKeys.clear();ui.run.textContent='Again';ui.status.textContent='';
 document.getElementById('winner').textContent=race.winner==='you'?'YOU WIN':race.winner==='flies'?'FLIES WIN':'PHOTO FINISH';
 victory.dataset.winner=race.winner;victory.hidden=false;document.getElementById('again').focus({preventScroll:true});
 if(playAudit){const wall=(performance.now()-playAudit.wall)/1000,sim=race.time-playAudit.sim;gpuDetails.textContent+=`\nLast uninterrupted play: ${wall.toFixed(2)} wall seconds → ${sim.toFixed(2)} simulation seconds (${(sim/wall).toFixed(2)}×).`;}
 if(recorder)captureFinishAt=t+1200;
}
function draw(t){
 requestAnimationFrame(draw);const dt=Math.max(0,(t-lastTime)/1000)||0;lastTime=t;
 if(running&&description&&!race.winner&&!raceReady){R.advance(race,description,heldKeys,raceDuration(dt));if(race.winner){race.winner=null;running=false;ui.run.textContent='Practice';}clock=0;}
 if(running&&description&&!race.winner&&raceReady){
  const last=frames.at(-1),flyFinish=(frames.find(f=>f.success)?.tick??Infinity)*description.dt;
  // Never convert slow compute into slow motion. An unexpected GPU deadline
  // miss is an explicit pause, then local preparation of the remaining race.
  const ceiling=ended?(last.success?last.tick:Infinity):Math.max(0,last.tick-2);
  const next=clock+raceDuration(dt)/description.dt;
  if(next>ceiling&&!ended){stop('GPU deadline missed · preparing remaining course');computeMode='prepared';rebuffering=true;ui.run.disabled=true;showGPU();}
  else R.advance(race,description,heldKeys,raceDuration(dt),flyFinish);
  clock=race.time/description.dt;
  while(frames.length>2&&frames[1].tick<=clock)frames.shift();
 }
 prefetch();
 // Idle preparation does not need sixty copies of an unchanged scene. Input
 // and race physics still update on every animation callback when playing.
 if(!running&&!recorder&&t-lastPaintTime<100)return;lastPaintTime=t;
 backdrop();if(!description)return;
 const a=frames[0],b=frames[1]||a,alpha=Math.min(1,Math.max(0,(clock-a.tick)/Math.max(1,b.tick-a.tick)));
 let q=C.interpolate(a,b,alpha);latest=alpha>=1?b:a;
 if(C.collides(q,description)){q=a.state;if(!renderFault){renderFault=true;stop('Render collision audit · Reset');}}
 announceWinner(t);
 recordButton.disabled=!recorder&&(!raceReady||!description||race.winner||!window.MediaRecorder);
 ui.passed.textContent=String(latest.stage).padStart(2,'0');
 // Freeze the cameras as well as both bodies at the finish line.
 if(!race.winner){camera+=(q[0]+.5-camera)*(1-Math.exp(-dt*6));playerCamera+=(race.state[0]+.5-playerCamera)*(1-Math.exp(-dt*6));}
 packetEvents=packetEvents.filter(e=>clock-e.at<3);
 if(raceReady)for(const frame of frames.slice(0,4))if(frame.tick>lastPacketTick&&frame.tick-1<=clock){packetEvents.push(...(frame.packets||[]));lastPacketTick=frame.tick;}
 const boxes=panels();
 drawWorld(playerMain?boxes.mini:boxes.main,q,camera,true,t);
 drawWorld(playerMain?boxes.main:boxes.mini,race.state,playerCamera,false,t);
 brainPanel(boxes.brain,a,b);
 if(recorder){
  // Recorded pixels identify actual model inference, not a teacher replay or
  // a claim that this experimental checkpoint has passed the final test set.
  ctx.fillStyle='#253471';ctx.font='800 18px "Segoe UI"';ctx.fillText('FLAPPY FLIES',28,48);
  ctx.fillStyle='#efdcff';ctx.font='10px "Segoe UI"';ctx.fillText('MALE CNS · 01 / 08',boxes.brain.x+14,boxes.brain.y+22);
  ctx.fillStyle='#a9a0d7';ctx.font='9px "Segoe UI"';ctx.fillText('165,122 neurons / fly',boxes.brain.x+14,boxes.brain.y+39);
  ctx.fillStyle='#276d8f';ctx.font='10px "Segoe UI"';ctx.fillText('8 fly neural cores × Sheaf-GBP',28,H-24);
  ctx.font='8px "Segoe UI"';ctx.fillText(`RECORDED MODEL INFERENCE · EXPERIMENTAL · ${captureIdentity.checkpoint}`,28,H-10);
  if(race.winner){ctx.fillStyle='#322354b0';ctx.fillRect(0,0,W,H);ctx.fillStyle='#fff6b1';ctx.textAlign='center';ctx.font=`900 ${Math.min(W*.1,90)}px "Segoe UI"`;ctx.fillText(document.getElementById('winner').textContent,W/2,H/2);ctx.textAlign='left';}
 }
 if(captureFinishAt!==null&&t>=captureFinishAt){captureFinishAt=null;finishCapture();}
}
async function init(){
 client.onProgress=p=>{if(p.phase==='adapter'){gpuInfo={adapter:p.adapter,candidates:p.candidates};showGPU();}else ui.status.textContent=`Fly brain ${Math.round(100*p.received/p.total)}% · practice now · cached next visit`;};
 client.onReady=async info=>{gpuInfo=info;showGPU();await reset();gpuBadge.classList.add('ready');};
 client.onError=message=>{client.ready=false;raceReady=false;stop(message+' · Practice is still available');ui.run.textContent='Practice';gpuBadge.textContent='CPU · Practice';gpuDetails.textContent=message+'\nPlayer physics runs locally. Neural racing is unavailable; no substitute AI or fake neural activity is shown.';};
 try{const data=await api('init');generation=data.generation;setDescription(data.description);}
 catch(e){ui.status.textContent=e.message;}
}
init();requestAnimationFrame(draw);
