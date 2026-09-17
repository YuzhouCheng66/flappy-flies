// No inference HTTP endpoint: every step executes in this visitor's worker.
const worker=new Worker(new URL('./inference-worker.mjs',import.meta.url),{type:'module'}),pending=new Map();let sequence=0;
export const client={ready:false,onProgress:()=>{},onReady:()=>{},onError:()=>{}};
worker.onmessage=({data})=>{if(data.event){if(data.event==='ready'){client.ready=true;client.onReady();}else if(data.event==='progress')client.onProgress(data);else client.onError(data.error);return;}const task=pending.get(data.id);if(!task)return;pending.delete(data.id);data.error?task.reject(Error(data.error)):task.resolve(data.result);};
worker.onerror=e=>{const error=Error(e.message||'The inference worker stopped');for(const p of pending.values())p.reject(error);pending.clear();client.onError(error.message);};
export function api(path,body={}){const type=path.split('/').pop();return new Promise((resolve,reject)=>{const id=++sequence;pending.set(id,{resolve,reject});worker.postMessage({id,type,body});});}
