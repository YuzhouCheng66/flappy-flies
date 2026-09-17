// Display level → immutable, audited source recording. Never relabel its hash.
export const COURSE_IDS=Object.freeze([
 1,10,
 21,24,28,32,36,40,
 41,44,47,50,53,56,60,
 61,64,67,70,73,76,80,
 81,84,87,90,93,96,98,100,
]);
export const COURSE_COUNT=COURSE_IDS.length;
export function sourceCourse(level){
 if(!Number.isInteger(level)||level<1||level>COURSE_COUNT)throw Error(`Choose a level from 1 to ${COURSE_COUNT}`);
 return COURSE_IDS[level-1];
}
export function previewLevel(search,hostname){
 if(!['localhost','127.0.0.1','[::1]'].includes(hostname))return null;
 const raw=new URLSearchParams(search).get('try'),n=Number(raw);
 return raw!==null&&Number.isInteger(n)&&n>=1&&n<=COURSE_COUNT?n:null;
}
