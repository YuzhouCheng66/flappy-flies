import {COURSE_COUNT} from './campaign-selection.mjs';
export function highestUnlocked(value){const n=Number(value);return Number.isInteger(n)&&n>=1&&n<=COURSE_COUNT?n:1;}
export function unlockAfterRace(unlocked,level,winner){
 const current=highestUnlocked(unlocked);
 if(winner!=='you'||!Number.isInteger(level)||level<1||level>current)return current;
 return Math.min(COURSE_COUNT,Math.max(current,level+1));
}
export function nextLevel(level,winner){return winner==='you'?Math.min(COURSE_COUNT,level+1):level;}
export function resumeLevel(saved,unlocked){const n=Number(saved);return Number.isInteger(n)&&n>=1&&n<=highestUnlocked(unlocked)?n:1;}
export function canEnter(level,unlocked){return Number.isInteger(level)&&level>=1&&level<=highestUnlocked(unlocked);}
