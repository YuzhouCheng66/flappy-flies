// Fixed-model campaign candidates. Acceptance requires an actual full rollout.
export function campaignLayout(level,attempt=0){
 if(!Number.isInteger(level)||level<1||level>100||!Number.isInteger(attempt)||attempt<0)throw Error('Invalid campaign candidate');
 const n=Math.ceil(level/20),p=(level-1)/99,seed=1000000+level*1000+attempt;
 let state=seed>>>0;const random=()=>{state=(state+0x6d2b79f5)>>>0;let t=Math.imul(state^(state>>>15),1|state);t^=t+Math.imul(t^(t>>>7),61|t);return((t^(t>>>14))>>>0)/4294967296;};
 const sign=1,amplitude=p<.25?0:.04+.12*(p-.25)/.75;
 const narrowCount=n===1?0:Math.min(Math.ceil(n/2),Math.floor((level-20)/18)+1);
 const indices=Array.from({length:n},(_,i)=>i).sort((a,b)=>(b%2-a%2)||b-a);
 const narrow=new Set(indices.slice(0,narrowCount)),spacing=2.35+.10*random();
 const wall_x=Array.from({length:n},(_,i)=>i*spacing);
 const gap_y=wall_x.map((_,i)=>sign*(-1)**i*amplitude*(.85+.15*random()));
 // Same .03 m reserve per edge as the published collision scene.
 const apertures=wall_x.map((_,i)=>narrow.has(i)?.90+.025*random():1.90+.04*(random()-.5));
 const initial=[-1.05+(random()-.5)*.012,gap_y[0]-.17+(random()-.5)*.024,-.04+(random()-.5)*.012,0,0,0];
 return{seed,level,attempt,difficulty:'campaign',wall_x,gap_y,apertures,initial,goals:wall_x.map((x,i)=>[x+1.05,gap_y[i],0]),player_dock_degrees:15-11.5*p,narrow_count:narrowCount};
}
export function campaignMeta(base,layout){
 const world_x=Math.max(base.config.world_x,layout.wall_x.at(-1)+2.25);
 return{...base,config:{...base.config,world_x},description:{...base.description,world_x,player_dock_angle:layout.player_dock_degrees*Math.PI/180,campaign_level:layout.level},layouts:{[layout.seed]:layout}};
}
// Optimistic geometric lower bound, NOT an executable or optimal policy.
// Horizontal speed is bounded by .40; at least dx-.03 must be traversed.
// Contact projection does not teleport or increase this velocity cap.
export function playerTimeLowerBound(description,settings){
 const q=description.state,goal=description.layout.goals.at(-1);
 const dx=Math.max(0,Math.abs(goal[0]-q[0])-.03),dy=goal[1]-q[1];
 const verticalCap=settings.speed+(dy<0?1:-1)*settings.gravity/settings.drag;
 const yaw=Math.abs(Math.atan2(Math.sin(goal[2]-q[2]),Math.cos(goal[2]-q[2])));
 return Math.max(dx/settings.speed,Math.max(0,Math.abs(dy)-.03)/verticalCap,Math.max(0,yaw-(description.player_dock_angle??settings.dockAngle))/settings.turnSpeed)+settings.dockHold;
}
