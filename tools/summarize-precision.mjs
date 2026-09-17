import fs from 'node:fs';
import {createHash} from 'node:crypto';
const root=new URL('../',import.meta.url);
const read=name=>fs.existsSync(new URL(name,root))?fs.readFileSync(new URL(name,root),'utf8').trim().split(/\r?\n/).filter(Boolean).map(s=>JSON.parse(s)):[];
const browser=read('local-results/browser.jsonl'),native=read('local-results/precision-native.jsonl');
const samples=[...browser,...native].filter(r=>r.type==='precision-parity');
const sourceFiles=['gpu-model.mjs','gpu-device.mjs','graph-layout.mjs','engine.mjs','tests/precision-audit.mjs'];
const hash=name=>createHash('sha256').update(fs.readFileSync(new URL(name,root))).digest('hex');
const report={generated:new Date().toISOString(),sourceSha256:Object.fromEntries(sourceFiles.map(name=>[name,hash(name)])),budgetMs:80/3.3,
 adapter:browser.filter(r=>r.type==='browser-adapter-comparison').at(-1),subgroupProbe:browser.filter(r=>r.type==='minimal-subgroup-probe').at(-1),
 timings:samples.map(r=>({runtime:r.runtime,time:r.time,auditId:r.auditId,kernel:r.kernel,precision:r.precision,memoryLayout:r.memoryLayout||'original',subgroupSize:r.subgroupSize||null,subgroupRows:r.subgroupRows||null,adapter:r.adapter,neuralTotalMs:r.neuralTotalMs,maxAbs:r.maxAbs})),
 rollouts:[...browser,...native].filter(r=>r.type==='precision-rollout'),
 browserCompileFailures:browser.filter(r=>r.type==='error'&&r.options?.subgroupSize),
 notes:['Native Dawn is not browser evidence.','80 recorded-input steps measure numerical error, not success rate.','Two-seed browser runs are smoke tests, not evidence for success rate >=90%.','No automatic precision switch or public deployment. No graph pruning or neural-update skipping.']};
fs.writeFileSync(new URL('local-results/precision-audit-summary.json',root),JSON.stringify(report,null,2));
console.log(JSON.stringify({timingRuns:report.timings.length,rollouts:report.rollouts.map(r=>({runtime:r.runtime,precision:r.precision,seed:r.seed,success:r.success,ticks:r.ticks})),compileFailures:report.browserCompileFailures.length},null,2));
