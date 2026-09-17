import {inspectAdapters} from './adapter-diagnostics.mjs';
onmessage=async()=>{try{postMessage(await inspectAdapters('dedicated-worker'));}catch(error){postMessage({error:error.message});}};
