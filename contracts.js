/* Shared by the live renderer and Node regression tests. No model emulation. */
(function(root){
  'use strict';
  function transform(p,q){const c=Math.cos(q[2]),s=Math.sin(q[2]);return[q[0]+c*p[0]-s*p[1],q[1]+s*p[0]+c*p[1]];}
  function intersects(a,b){
    for(const poly of [a,b])for(let i=0;i<poly.length;i++){
      const p=poly[i],q=poly[(i+1)%poly.length],nx=p[1]-q[1],ny=q[0]-p[0];
      const pa=a.map(v=>v[0]*nx+v[1]*ny),pb=b.map(v=>v[0]*nx+v[1]*ny);
      if(Math.max(...pa)<Math.min(...pb)||Math.max(...pb)<Math.min(...pa))return false;
    }return true;
  }
  function collides(q,d){
    if(!q.every(Number.isFinite))return true;
    return d.rectangles.some(rect=>{
      const points=rect.map(p=>transform(p,q));
      return points.some(p=>Math.abs(p[0])>d.world_x||Math.abs(p[1])>d.world_y)||
        d.colliders.some(wall=>intersects(points,wall));
    });
  }
  function interpolate(a,b,t){
    if(a!==b&&b.tick!==a.tick+1)throw Error('Nonconsecutive physics frames');
    const q=a.state.map((v,i)=>v+(b.state[i]-v)*t);
    const yaw=Math.atan2(Math.sin(b.state[2]-a.state[2]),Math.cos(b.state[2]-a.state[2]));
    q[2]=a.state[2]+yaw*t;return q;
  }
  function validateBatch(after,frames,d){
    let tick=after;
    for(const f of frames){
      if(f.tick!==++tick)throw Error('Physics stream interrupted · Reset');
      if(collides(f.state,d))throw Error('Collision audit stopped the stream');
    }
    return tick;
  }
  // All solid artwork uses one common orthographic extrusion. Positive screen
  // y makes the upper wall's lower depth edge slope down and right.
  function extrusionDepth(scale){return[.060*scale,.046*scale];}
  function tOutline(rectangles){
    const bounds=rectangles.map(r=>({xmin:Math.min(...r.map(p=>p[0])),xmax:Math.max(...r.map(p=>p[0])),ymin:Math.min(...r.map(p=>p[1])),ymax:Math.max(...r.map(p=>p[1]))}));
    if(bounds.length!==2)throw Error('The apple socket requires the exported two-box T');
    bounds.sort((a,b)=>(b.xmax-b.xmin)-(a.xmax-a.xmin));const [cap,stem]=bounds;
    return[[cap.xmin,cap.ymax],[cap.xmax,cap.ymax],[cap.xmax,cap.ymin],[stem.xmax,cap.ymin],[stem.xmax,stem.ymin],[stem.xmin,stem.ymin],[stem.xmin,cap.ymin],[cap.xmin,cap.ymin]];
  }
  const api={transform,intersects,collides,interpolate,validateBatch,extrusionDepth,tOutline};
  if(typeof module==='object')module.exports=api;else root.CargoContracts=api;
})(typeof globalThis!=='undefined'?globalThis:this);
