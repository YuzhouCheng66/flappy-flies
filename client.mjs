// Campaign is the default. Full local inference remains an explicit mode.
import {campaignAPI} from './campaign-client.mjs';
// Research-only preview: never silently substitute a different model on the
// public site. This route is restricted to loopback and labelled in the UI.
const candidate=['localhost','127.0.0.1','[::1]'].includes(location.hostname)?new URLSearchParams(location.search).get('candidate'):null;
const candidateHash=/^[a-f0-9]{12}$/.test(candidate||'')?candidate:null;
const workerURL=new URL('./inference-worker.mjs',import.meta.url);if(candidateHash)workerURL.searchParams.set('candidate',candidateHash);
const campaign=!candidateHash&&new URLSearchParams(location.search).get('mode')!=='live';
const worker=campaign?null:new Worker(workerURL,{type:'module'}),pending=new Map();let sequence=0;
export const client={campaign,candidate:candidateHash,ready:campaign,onProgress:()=>{},onReady:()=>{},onError:()=>{}};
if(worker){worker.onmessage=({data})=>{if(data.event){if(data.event==='ready'){client.ready=true;client.onReady(data);}else if(data.event==='progress')client.onProgress(data);else client.onError(data.error);return;}const task=pending.get(data.id);if(!task)return;pending.delete(data.id);data.error?task.reject(Error(data.error)):task.resolve(data.result);};
worker.onerror=e=>{const error=Error(e.message||'The inference worker stopped');for(const p of pending.values())p.reject(error);pending.clear();client.onError(error.message);};}
export function api(path,body={}){if(campaign)return campaignAPI(path,body);const type=path.split('/').pop();return new Promise((resolve,reject)=>{const id=++sequence;pending.set(id,{resolve,reject});worker.postMessage({id,type,body});});}
