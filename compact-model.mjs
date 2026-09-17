// Explicit compact research candidate. Never presented as full-connectome activity.
export class CompactModel{
 constructor(meta){this.meta=meta;this.hidden=Array.from({length:8},()=>new Float32Array(128));this.kernel='compact-local-GRU-distilled-v1';this.parameters=Object.fromEntries(Object.entries(meta.parameters).map(([k,v])=>[k,new Float32Array(v.data)]));}
 reset(){this.hidden.forEach(h=>h.fill(0));}
 linear(name,x,out,activate=false){const w=this.parameters[name+'.weight'],b=this.parameters[name+'.bias'];for(let r=0;r<out.length;r++){let s=b[r];for(let j=0;j<x.length;j++)s+=w[r*x.length+j]*x[j];out[r]=activate?s/(1+Math.exp(-s)):s;}return out;}
 agent(normalized,a){
  const e=this.linear('embed.0',normalized,new Float32Array(128),true),h=this.hidden[a],next=new Float32Array(128),p=this.parameters;
  for(let u=0;u<128;u++){const iv=[],hv=[];for(let g=0;g<3;g++){const r=g*128+u;let si=p['gru.bias_ih_l0'][r],sh=p['gru.bias_hh_l0'][r];for(let j=0;j<128;j++){si+=p['gru.weight_ih_l0'][r*128+j]*e[j];sh+=p['gru.weight_hh_l0'][r*128+j]*h[j];}iv[g]=si;hv[g]=sh;}const reset=1/(1+Math.exp(-iv[0]-hv[0])),update=1/(1+Math.exp(-iv[1]-hv[1]));next[u]=(1-update)*Math.tanh(iv[2]+reset*hv[2])+update*h[u];}
  this.hidden[a]=next;return this.linear('head.2',this.linear('head.0',next,new Float32Array(128),true),new Float32Array(6));
 }
 async step(observations){const start=performance.now(),twist=[],precision=[];for(let a=0;a<8;a++){
  const m=this.meta,n=observations[a].map((x,i)=>Math.max(-8,Math.min(8,(x-m.xmean[i])/m.xstd[i]))),out=this.agent(n,a).map((x,i)=>x*m.ystd[i]+m.ymean[i]);
  const v=[out[0]*.65,out[1]*.65,out[2]*1.4],ratio=Math.max(1,Math.hypot(v[0],v[1])/.18,Math.abs(v[2])/.4);twist.push(Array.from(out.slice(0,3),x=>Math.fround(x/ratio)));precision.push(Array.from(out.slice(3),x=>Math.fround(Math.exp(Math.max(-3,Math.min(3,x))))));
 }this.timing={wall:performance.now()-start};return{twist,precision,brain:Array.from(this.hidden[0]),brain_rms:this.hidden.map(h=>Math.sqrt(h.reduce((s,x)=>s+x*x,0)/h.length)),features:[]};}
}
