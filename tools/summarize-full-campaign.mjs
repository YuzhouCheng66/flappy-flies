import fs from 'node:fs';
const root=new URL('../local-results/full-campaign-training/',import.meta.url),cp=new URL('checkpoints/',root);
const records=fs.readdirSync(root).filter(n=>/^r\d+-.*\.json$/.test(n)).map(n=>JSON.parse(fs.readFileSync(new URL(n,root))));
const experiments=fs.readdirSync(cp).filter(n=>n.endsWith('.eval.json')).map(n=>{
 const evaluation=JSON.parse(fs.readFileSync(new URL(n,cp))),old=evaluation.rows.filter(r=>r.level<=60),hard=evaluation.rows.filter(r=>r.level>60);
 return{file:n,weights:evaluation.model,successes:evaluation.successes,total:evaluation.total,levelIds:evaluation.rows.map(r=>r.level),oldSuccesses:old.filter(r=>r.success).length,oldTested:old.length,hardSuccesses:hard.filter(r=>r.success).length,hardTested:hard.length,old60RegressionPassed:new Set(old.filter(r=>r.success).map(r=>r.level)).size===60};
});
const summary={generatedAt:new Date().toISOString(),data:{episodes:records.length,studentEpisodes:records.filter(r=>r.mode==='student').length,guidedEpisodes:records.filter(r=>r.mode==='guided').length,recoveryEpisodes:records.filter(r=>r.mode==='recovery').length,agentRows:records.reduce((s,r)=>s+r.ticks*8,0)},experiments,interpretation:'Guided/recovery successes are not model successes. Screening mixes training-level regression and validation; it is not a held-out success-rate claim. No release approval is implied.'};
fs.writeFileSync(new URL('audit-summary.json',root),JSON.stringify(summary,null,2));console.log(JSON.stringify(summary,null,2));
