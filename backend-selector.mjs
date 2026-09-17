// Workload-based selection; no GPU-brand or CPU-core-count performance promise.
// Factories own cleanup. A candidate is never accepted just because it exists.
export async function measureBackend(model,observations,{warmup=3,samples=8}={}){
 model.reset();for(let i=0;i<warmup;i++)await model.step(observations[i%observations.length]);
 const timings=[];for(let i=0;i<samples;i++){const start=performance.now();await model.step(observations[(i+warmup)%observations.length]);timings.push(performance.now()-start);}
 model.reset();const ordered=[...timings].sort((a,b)=>a-b);return{medianMs:ordered[Math.floor(ordered.length/2)],p95Ms:ordered[Math.ceil(ordered.length*.95)-1],samples:timings};
}
export function chooseMeasuredBackend(measurements,{medianBudgetMs=15,p95BudgetMs=20}={}){
 return measurements.filter(m=>m.validated&&Number.isFinite(m.medianMs)&&m.medianMs>0&&Number.isFinite(m.p95Ms)&&m.p95Ms>0&&m.medianMs<=medianBudgetMs&&m.p95Ms<=p95BudgetMs).sort((a,b)=>a.p95Ms-b.p95Ms||a.medianMs-b.medianMs)[0]||null;
}
