// Literal local27 / analytic 16->3 GBP / bounded force physics port.
// Float32 operations mirror the trained simulator; no teacher or trajectory.
export const F=Math.fround;
const A=(a,b)=>F(F(a)+F(b)),S=(a,b)=>F(F(a)-F(b)),M=(a,b)=>F(F(a)*F(b)),D=(a,b)=>F(F(a)/F(b));
const clamp=(v,lo,hi)=>Math.max(lo,Math.min(hi,v)),wrap=v=>F(Math.atan2(F(Math.sin(v)),F(Math.cos(v))));
const norm=(x,y)=>F(Math.sqrt(A(M(x,x),M(y,y))));
export function transform(p,q){const c=F(Math.cos(q[2])),s=F(Math.sin(q[2]));return[A(S(M(c,p[0]),M(s,p[1])),q[0]),A(A(M(s,p[0]),M(c,p[1])),q[1])];}
function wallsFor(layout,cfg){const walls=[];layout.wall_x.forEach((x,i)=>{x=F(x);const half=Math.max(0,S(D(layout.apertures[i],2),.03)),y=F(layout.gap_y[i]),x0=S(x,cfg.wall_slab),x1=A(x,cfg.wall_slab);for(const [low,high] of [[F(-cfg.world_y),S(y,half)],[A(y,half),F(cfg.world_y)]])if(high>low)walls.push([[x0,low],[x1,low],[x1,high],[x0,high]]);});return walls;}
export function clearance(q,walls,meta){
 let margin=Infinity,mx=0,my=0;
 for(const rect of meta.description.rectangles){
  const points=rect.map(p=>transform(p,q)),axes=[];
  for(let j=0;j<2;j++){const x=S(points[j+1][0],points[j][0]),y=S(points[j+1][1],points[j][1]),length=Math.max(1e-8,norm(x,y));axes.push([D(-y,length),D(x,length)]);}
  axes.push([1,0],[0,1]);
  for(const p of points){mx=Math.max(mx,Math.abs(p[0]));my=Math.max(my,Math.abs(p[1]));}
  for(const wall of walls){let pair=-Infinity;
   for(const [x,y] of axes){const body=points.map(p=>A(M(p[0],x),M(p[1],y))),obstacle=wall.map(p=>A(M(p[0],x),M(p[1],y)));pair=Math.max(pair,S(Math.min(...obstacle),Math.max(...body)),S(Math.min(...body),Math.max(...obstacle)));}
   margin=Math.min(margin,pair);
  }
 }
 let boundary=Math.min(S(meta.config.world_x,mx),S(meta.config.world_y,my));if(boundary===0)boundary=1.1754943508222875e-38;
 return Math.min(margin,boundary);
}
export function observe(state,layout,stage,context,meta){
 const cfg=meta.config,handles=meta.description.handles,s=F(Math.sin(state[2])),c=F(Math.cos(state[2])),goal=layout.goals[stage].map(F);
 const rear=Math.min(...meta.description.rectangles.flat().map(p=>transform(p,state)[0]));
 return handles.map(handle=>{
  const world=transform(handle,state),offset=world.map((v,i)=>S(v,state[i]));
  const velocity=[S(state[3],M(state[5],offset[1])),A(state[4],M(state[5],offset[0]))];
  const visible=layout.wall_x.map(x=>Math.abs(S(x,world[0]))<=cfg.wall_sensor_range&&A(A(x,cfg.wall_slab),.03)>=rear);
  const order=layout.wall_x.map((x,i)=>({i,score:visible[i]?Math.abs(S(x,S(goal[0],1.05))):Infinity})).sort((a,b)=>a.score-b.score||a.i-b.i).slice(0,2).map(p=>p.i);
  const vis=order.map(i=>+visible[i]),rx=order.map((i,k)=>M(D(S(layout.wall_x[i],world[0]),cfg.wall_sensor_range),vis[k])),ry=order.map((i,k)=>M(D(S(layout.gap_y[i],world[1]),cfg.world_y),vis[k])),half=order.map((i,k)=>M(D(D(layout.apertures[i],2),cfg.world_y),vis[k]));
  const gx=D(S(goal[0],world[0]),2.6),dir=Math.sign(gx);let numerator=0,denominator=0;
  for(let k=0;k<2;k++){const x=M(M(dir,rx[k]),cfg.wall_sensor_range),sig=D(1,A(1,F(Math.exp(-M(12,A(x,.22))))));const w=M(M(vis[k],sig),F(Math.exp(M(-1.5,Math.abs(x)))));denominator=A(denominator,w);numerator=A(numerator,M(w,D(M(ry[k],cfg.world_y),2.6)));}
  const base=[s,c,...handle,D(world[0],2.6),D(world[1],cfg.world_y),...velocity.map(v=>D(v,2)),gx,D(numerator,Math.max(1e-6,denominator)),...rx,...ry,...half,...vis];
  // Reconstruct only this handle's locally visible wall records, matching
  // the float32 finite-difference interface used for V17 training.
  const contact=[M(base[4],2.6),M(base[5],cfg.world_y)],centre=[S(contact[0],S(M(c,handle[0]),M(s,handle[1]))),S(contact[1],A(M(s,handle[0]),M(c,handle[1])))],theta=F(Math.atan2(s,c));
  let feedback=[0,0,0,0],best=Infinity;
  for(let k=0;k<2;k++)if(vis[k]){
   const x=A(contact[0],M(rx[k],cfg.wall_sensor_range)),y=A(contact[1],M(ry[k],cfg.world_y)),h=M(half[k],cfg.world_y);
   const walls=wallsFor({wall_x:[x],gap_y:[y],apertures:[M(h,2)]},cfg);
   const at=(dy=0,da=0,dx=0)=>clearance([A(centre[0],dx),A(centre[1],dy),A(theta,da)],walls,meta);
   const value=at();if(value<best){best=value;feedback=[D(value,.001),D(S(at(.0001),at(-.0001)),.0002),D(S(at(0,.0001),at(0,-.0001)),.0002),D(S(at(M(M(base[7],2),cfg.dt),0,M(M(base[6],2),cfg.dt)),value),.001)].map(v=>clamp(v,-32,32));}
  }
  base[4]=S(base[4],D(S(goal[0],1.05),2.6));base[5]=S(base[5],D(goal[1],cfg.world_y));
  const index=handles.indexOf(handle);return[...base,...feedback,+context.hit,...context.force[index],...context.rejected[index]];
 });
}
function solve(matrix,rhs){const a=matrix.map((r,i)=>[...r,rhs[i]]);for(let k=0;k<3;k++){let pivot=k;for(let i=k+1;i<3;i++)if(Math.abs(a[i][k])>Math.abs(a[pivot][k]))pivot=i;[a[k],a[pivot]]=[a[pivot],a[k]];for(let i=k+1;i<3;i++){const f=D(a[i][k],a[k][k]);for(let j=k;j<4;j++)a[i][j]=S(a[i][j],M(f,a[k][j]));}}const x=[0,0,0];for(let i=2;i>=0;i--){let value=a[i][3];for(let j=i+1;j<3;j++)value=S(value,M(a[i][j],x[j]));x[i]=D(value,a[i][i]);}return x;}
function sums(values){return values.reduce((a,v)=>A(a,v),0);}
export function twistToWrench(twist,obs,meta){
 const cfg=meta.config,handles=meta.description.handles,bx=sums(handles.map(p=>p[0])),by=sums(handles.map(p=>p[1])),r2=sums(handles.flatMap(p=>p.map(x=>M(x,x))));
 const normal=[[8,0,-by],[0,8,bx],[-by,bx,r2]];
 return obs.map((o,i)=>{const s=o[0],c=o[1],vx=M(o[6],2),vy=M(o[7],2),body=[A(M(c,vx),M(s,vy)),A(M(-s,vx),M(c,vy))];const current=solve(normal,[...body,A(M(-o[3],body[0]),M(o[2],body[1]))]).map(x=>M(8,x));
  const desired=twist[i],f=[0,1].map(j=>D(M(cfg.mass,S(D(M(desired[j],.65),1-cfg.linear_damping*cfg.dt),current[j])),cfg.dt));
  const torque=D(M(cfg.inertia,S(D(M(desired[2],1.4),1-cfg.angular_damping*cfg.dt),current[2])),cfg.dt);
  return[D(S(M(c,f[0]),M(s,f[1])),32),D(A(M(s,f[0]),M(c,f[1])),32),D(torque,24)];
 });
}
export function coordinate(mean,precision,obs,meta,stats=null){
 const {edges,tree_mask:tree,gbp}=meta,sender=edges.flat(),receiver=sender.map((_,i)=>sender[i^1]),count=sender.length;
 const zero=()=>Array.from({length:count},()=>[0,0,0]);let omega=zero(),xi=zero();const qs=mean.map((v,i)=>v.map((x,k)=>M(x,precision[i][k]))),traces=[],ptraces=[];
 const incoming=()=>{const q=precision.map(r=>r.slice()),h=qs.map(r=>r.slice());for(let j=0;j<count;j++)for(let k=0;k<3;k++){q[sender[j]][k]=A(q[sender[j]][k],omega[j^1][k]);h[sender[j]][k]=A(h[sender[j]][k],xi[j^1][k]);}return[q,h];};
 for(let sweep=0;sweep<gbp.steps;sweep++){
  const [q,h]=incoming(),nw=zero(),nx=zero();
  for(let j=0;j<count;j++)for(let k=0;k<3;k++){
   const v=D(1,Math.max(gbp.eps,S(q[sender[j]][k],omega[j^1][k]))),mu=M(v,S(h[sender[j]][k],xi[j^1][k]));
   const target=Math.min(gbp.max_precision,D(1,Math.max(gbp.eps,A(tree[j>>1]?gbp.sigma2:1e6,v))));
   nw[j][k]=gbp.damping===1?target:A(omega[j][k],M(gbp.damping,S(target,omega[j][k])));nx[j][k]=gbp.damping===1?M(target,mu):A(xi[j][k],M(gbp.damping,S(M(target,mu),xi[j][k])));
  }omega=nw;xi=nx;traces.push(xi);ptraces.push(omega);
 }
 const [q,h]=incoming(),section=h.map((v,i)=>v.map((x,k)=>D(x,Math.max(q[i][k],gbp.eps))));let setup=[];
 if(!stats){const te=edges.filter((_,i)=>tree[i]),source=[...te.map(e=>e[0]),...te.map(e=>e[1])],destination=[...te.map(e=>e[1]),...te.map(e=>e[0])],local=obs.map(o=>[1,o[2],o[3],A(M(o[2],o[2]),M(o[3],o[3]))]);let messages=Array.from({length:14},()=>[0,0,0,0]);
  for(let sweep=0;sweep<7;sweep++){const inc=Array.from({length:8},()=>[0,0,0,0]);for(let j=0;j<14;j++)for(let k=0;k<4;k++)inc[destination[j]][k]=A(inc[destination[j]][k],messages[j][k]);messages=source.map((s,j)=>local[s].map((v,k)=>S(A(v,inc[s][k]),messages[(j+7)%14][k])));setup.push(messages);}
  stats=local.map(v=>v.slice());for(let j=0;j<14;j++)for(let k=0;k<4;k++)stats[destination[j]][k]=A(stats[destination[j]][k],messages[j][k]);
 }
 const forces=obs.map((o,i)=>{const [n,bx,by,r2]=stats[i],s=o[0],c=o[1],x=S(M(c,o[2]),M(s,o[3])),y=A(M(s,o[2]),M(c,o[3])),sx=S(M(c,bx),M(s,by)),sy=A(M(s,bx),M(c,by));
  const dual=solve([[A(n,1e-5),0,-sy],[0,A(n,1e-5),sx],[-sy,sx,A(r2,1e-5)]],section[i].map((v,k)=>M(v,M(k===2?.75:1,M(n,meta.config.max_force)))));
  const force=[S(dual[0],M(y,dual[2])),A(dual[1],M(x,dual[2]))],scale=Math.min(1,D(meta.config.max_force,Math.max(1e-6,norm(...force))));return force.map(v=>M(v,scale));
 });return{forces,stats,section,sender,receiver,natural:traces,precision:ptraces,setup};
}
export function dynamics(q,forces,meta){
 const c=meta.config,r=meta.description.handles.map(p=>transform(p,[0,0,q[2]]));
 const fx=sums(forces.map(p=>p[0])),fy=sums(forces.map(p=>p[1])),torque=sums(forces.map((p,i)=>S(M(r[i][0],p[1]),M(r[i][1],p[0]))));
 const v=[fx,fy].map((force,i)=>M(A(q[3+i],D(M(c.dt,force),c.mass)),1-c.linear_damping*c.dt)),omega=M(A(q[5],D(M(c.dt,torque),c.inertia)),1-c.angular_damping*c.dt);
 return[A(q[0],M(c.dt,v[0])),A(q[1],M(c.dt,v[1])),wrap(A(q[2],M(c.dt,omega))),...v,omega];
}
export function transition(q,forces,walls,meta){
 const attempted=dynamics(q,forces,meta),delta=attempted.map((v,i)=>S(v,q[i]));delta[2]=wrap(delta[2]);const at=t=>q.map((v,i)=>A(v,M(t,delta[i])));
 let hit=false;for(let i=1;i<=meta.config.collision_substeps;i++){const pose=at(i/meta.config.collision_substeps);pose[2]=wrap(pose[2]);if(clearance(pose,walls,meta)<=1e-6){hit=true;break;}}
 if(!hit){const radius=Math.max(...meta.description.rectangles.flat().map(p=>Math.hypot(...p))),motion=norm(delta[0],delta[1])+radius*Math.abs(delta[2]);
  const certify=(centre,half,depth)=>{const margin=clearance(at(centre),walls,meta);if(margin-motion*half>2e-6)return true;if(margin<=2e-6||depth===16)return false;return certify(centre-half/2,half/2,depth+1)&&certify(centre+half/2,half/2,depth+1);};hit=!certify(.5,.5,0);
 }
 const state=hit?[...q.slice(0,3),0,0,0]:attempted;
 const rejected=meta.description.handles.map(p=>{const a=transform(p,attempted),b=transform(p,state);return a.map((v,i)=>hit?D(S(v,b[i]),meta.config.dt*2):0);});
 return{state,hit,attempted,context:{hit,force:forces.map(p=>p.map(v=>D(v,meta.config.max_force))),rejected}};
}
export function layoutFor(seed,meta){
 if(meta.layouts[String(seed)])return structuredClone(meta.layouts[String(seed)]);
 // New seeds use a documented portable PRNG; ranges are identical to Twist.
 let s=seed>>>0;const rng=()=>{s=(s+0x6d2b79f5)>>>0;let t=Math.imul(s^(s>>>15),1|s);t^=t+Math.imul(t^(t>>>7),61|t);return((t^(t>>>14))>>>0)/4294967296;};
 const apertures=Array.from({length:4},()=>.90+.04*rng()),sign=rng()<.5?-1:1,gap_y=Array.from({length:4},(_,i)=>sign*(-1)**i*(.09+.09*rng())),wall_x=[0,2.4,4.8,7.2],initial=[F(-1.05),F(gap_y[0]),0,0,0,0];for(let i=0;i<3;i++)initial[i]=A(initial[i],rng()*.024-.012);
 return{seed,difficulty:'rotation',apertures,gap_y,wall_x,initial,goals:wall_x.map((x,i)=>[x+1.05,gap_y[i],0])};
}
export class Engine{
 constructor(meta,gpu){this.meta=meta;this.gpu=gpu;this.reset(91000);}
 reset(seed){this.layout=layoutFor(seed,this.meta);this.walls=wallsFor(this.layout,this.meta.config);this.state=this.layout.initial.map(F);this.stage=this.tick=this.stable=this.contacts=this.stageStart=this.wireBytes=0;this.done=this.success=false;this.stats=null;this.context={hit:false,force:Array.from({length:8},()=>[0,0]),rejected:Array.from({length:8},()=>[0,0])};this.gpu?.reset();return this.description();}
 description(){return{...this.meta.description,layout:this.layout,colliders:this.walls,state:this.state,snapshot:this.snapshot()};}
 snapshot(){return{tick:this.tick,state:this.state,stage:this.stage,goal:this.layout.goals[Math.min(3,this.stage)],done:this.done,success:this.success,contacts:this.contacts};}
 async step(){if(this.done)return null;const obs=observe(this.state,this.layout,this.stage,this.context,this.meta),neural=await this.gpu.step(obs),mean=twistToWrench(neural.twist,obs,this.meta),gbp=coordinate(mean,neural.precision,obs,this.meta,this.stats);this.stats=gbp.stats;
  const trans=transition(this.state,gbp.forces,this.walls,this.meta);this.state=trans.state;this.context=trans.context;this.tick++;this.contacts+=+trans.hit;
  if(clearance(this.state,this.walls,this.meta)<=0)throw Error('Collision authority rejected an accepted state');
  const goal=this.layout.goals[this.stage],q=this.state,atGoal=Math.hypot(q[0]-goal[0],q[1]-goal[1])<=.03&&Math.abs(wrap(q[2]-goal[2]))<=.05&&Math.hypot(q[3],q[4])<=.05&&Math.abs(q[5])<=.08;
  this.stable=atGoal?this.stable+1:0;let changed=false;if(this.stable>=5){this.stage++;this.stable=0;changed=true;if(this.stage===4)this.done=this.success=true;else this.stageStart=this.tick;}
  if(this.tick-this.stageStart>=384)this.done=true;
  this.wireBytes+=gbp.sender.length*3*2*4*4+gbp.setup.length*14*4*4;
  this.last={obs,features:neural.features,mean,precision:neural.precision,forces:gbp.forces};
  return{...this.snapshot(),hit:trans.hit,stage_changed:changed,sender:gbp.sender,receiver:gbp.receiver,natural:gbp.natural,precision:gbp.precision,forces:gbp.forces,wire_bytes:this.wireBytes,brain:neural.brain,brain_rms:neural.brain_rms};
 }
}
