/* Browser/Node shared arcade physics. No checkpoint, teacher or fly emulation. */
(function(root){
 'use strict';
 const C=typeof module==='object'?require('./contracts.js'):root.CargoContracts;
 const KEYS=new Set(['KeyW','KeyA','KeyS','KeyD','ArrowUp','ArrowLeft','ArrowDown','ArrowRight','KeyQ','KeyE']);
 const SETTINGS=Object.freeze({gravity:.24,speed:.40,turnSpeed:1.1,drag:8,turnDrag:10,step:1/120,dockHold:.4});
 const angle=x=>Math.atan2(Math.sin(x),Math.cos(x));
 function inputs(keys){
  let x=Number(keys.has('KeyD')||keys.has('ArrowRight'))-Number(keys.has('KeyA')||keys.has('ArrowLeft'));
  let y=Number(keys.has('KeyW')||keys.has('ArrowUp'))-Number(keys.has('KeyS')||keys.has('ArrowDown'));
  const n=Math.max(1,Math.hypot(x,y));
  return[x/n,y/n,Number(keys.has('KeyQ'))-Number(keys.has('KeyE'))];
 }
 function create(d){return{state:[...d.state],time:d.snapshot.tick*d.dt,winner:null,finishedAt:null,dock:0,stage:0,contacts:0};}
 // A separating axis at the midpoint certifies the WHOLE motion interval if
 // its gap exceeds a bound on every rotating/translating body's displacement.
 // Ambiguous intervals are bisected; unresolved intervals are rejected, never
 // assumed safe. This is conservative floating-point SAT, not exact arithmetic.
 function separated(a,b,margin){
  for(const poly of [a,b])for(let i=0;i<poly.length;i++){
   const p=poly[i],q=poly[(i+1)%poly.length],nx=p[1]-q[1],ny=q[0]-p[0],n=Math.hypot(nx,ny);
   const pa=a.map(v=>(v[0]*nx+v[1]*ny)/n),pb=b.map(v=>(v[0]*nx+v[1]*ny)/n);
   if(Math.max(...pa)+margin<Math.min(...pb)||Math.max(...pb)+margin<Math.min(...pa))return true;
  }return false;
 }
 function safeMotion(a,b,d,depth=0){
  if(C.collides(a,d)||C.collides(b,d))return false;
  const mid=a.map((v,i)=>(v+b[i])/2),radius=Math.max(...d.rectangles.flat().map(p=>Math.hypot(...p)));
  const margin=Math.hypot(b[0]-a[0],b[1]-a[1])/2+radius*Math.abs(b[2]-a[2])/2+1e-10;
  const certified=d.rectangles.every(rect=>{
   const p=rect.map(v=>C.transform(v,mid));
   return p.every(v=>Math.abs(v[0])+margin<d.world_x&&Math.abs(v[1])+margin<d.world_y)&&d.colliders.every(w=>separated(p,w,margin));
  });
  if(certified)return true;
  if(depth>=12||C.collides(mid,d))return false;
  return safeMotion(a,mid,d,depth+1)&&safeMotion(mid,b,d,depth+1);
 }
 function playerStep(r,d,keys,dt){
  const input=inputs(keys),q=r.state,s=SETTINGS;
  // Closed-form damped velocity; input has no dependency on viewport size.
  for(let i=0;i<3;i++){
   const drag=i===2?s.turnDrag:s.drag,desired=input[i]*(i===2?s.turnSpeed:s.speed)-(i===1?s.gravity/drag:0);
   q[i+3]=desired+(q[i+3]-desired)*Math.exp(-drag*dt);
  }
  let target=q.slice();for(let i=0;i<3;i++)target[i]+=q[i+3]*dt;
  if(safeMotion(q,target,d))r.state=target;
  else{
   r.contacts++;
   // Sliding is allowed, penetration is not. Each accepted axis segment is
   // independently swept-certified, including rotation of both T rectangles.
   for(const i of [2,0,1]){target=r.state.slice();target[i]+=r.state[i+3]*dt;if(safeMotion(r.state,target,d))r.state=target;else r.state[i+3]=0;}
  }
  r.state[2]=angle(r.state[2]);
  const goal=d.layout.goals.at(-1),v=r.state;
  const near=Math.hypot(v[0]-goal[0],v[1]-goal[1])<=.03&&Math.abs(angle(v[2]-goal[2]))<=.05&&Math.hypot(v[3],v[4])<=.05&&Math.abs(v[5])<=.08;
  r.dock=near?r.dock+dt:0;
  const rear=Math.min(...d.rectangles.flat().map(p=>C.transform(p,v)[0]));
  r.stage=d.layout.wall_x.filter((_,i)=>rear>Math.max(...d.colliders[2*i].map(p=>p[0]))).length;
 }
 function finish(r,winner,time){if(!r.winner){r.winner=winner;r.finishedAt=time;r.time=time;}return r.winner;}
 // Both contestants consume this single displayed simulation clock. No player
 // movement while paused/buffering, no awarding a prefetched future fly win.
 function advance(r,d,keys,duration,flyFinish=Infinity){
  if(r.winner||duration<=0)return;
  const end=Math.min(r.time+duration,flyFinish);
  while(r.time<end-1e-10){
   const dt=Math.min(SETTINGS.step,end-r.time,r.dock>0?Math.max(1e-9,SETTINGS.dockHold-r.dock):Infinity);playerStep(r,d,keys,dt);r.time+=dt;
   if(r.dock>=SETTINGS.dockHold-1e-9){finish(r,Math.abs(r.time-flyFinish)<1e-9?'draw':'you',r.time);return;}
  }
  if(flyFinish<=r.time+1e-9)finish(r,'flies',flyFinish);
 }
 const api={KEYS,SETTINGS,inputs,create,safeMotion,playerStep,advance,finish};
 if(typeof module==='object')module.exports=api;else root.CargoRace=api;
})(typeof globalThis!=='undefined'?globalThis:this);
