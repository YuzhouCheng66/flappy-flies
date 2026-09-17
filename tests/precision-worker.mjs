import {auditPrecision} from './precision-audit.mjs';
let busy=false;
onmessage=async({data})=>{if(busy)return;busy=true;try{for(const options of data.suite||[data]){try{await auditPrecision(options,row=>postMessage(row));}catch(error){postMessage({type:'error',options,message:error.message,stack:error.stack});}}}finally{busy=false;postMessage({type:'idle'});}};
