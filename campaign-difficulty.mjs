// Arcade difficulty only; never changes the recorded neural policy or physics.
export function campaignDifficulty(level,total=30){
 const p=(Math.min(total,Math.max(1,level))-1)/Math.max(1,total-1),lerp=(a,b)=>a+(b-a)*p;
 return{
  player_dock_radius:lerp(.18,.03),
  player_dock_angle:lerp(30,3.5)*Math.PI/180,
  player_dock_speed:lerp(.30,.05),
  player_dock_turn_speed:lerp(.40,.08),
  player_dock_hold:lerp(.10,.40),
  fly_speed_factor:lerp(.80,1),
 };
}
