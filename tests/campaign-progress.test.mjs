import test from 'node:test';
import assert from 'node:assert/strict';
import {highestUnlocked,unlockAfterRace,nextLevel,resumeLevel,canEnter} from '../campaign-progress.mjs';
test('Only a human win unlocks the next level',()=>{
 for(const outcome of ['flies','draw',null]){assert.equal(unlockAfterRace(1,1,outcome),1);assert.equal(nextLevel(1,outcome),1);}
 assert.equal(unlockAfterRace(1,1,'you'),2);assert.equal(nextLevel(1,'you'),2);
 assert.equal(unlockAfterRace(8,1,'you'),8);assert.equal(unlockAfterRace(1,20,'you'),1);
 assert.equal(unlockAfterRace(30,30,'you'),30);assert.equal(nextLevel(30,'you'),30);
 for(const bad of [null,'bad',0,31,1.5])assert.equal(highestUnlocked(bad),1);
 assert.equal(highestUnlocked('20'),20);
});
test('Retry/resume stays on the current level and every earlier level is legal',()=>{
 for(let x=1;x<=30;x++){
  assert.equal(nextLevel(x,'flies'),x);assert.equal(resumeLevel(String(x),x),x);
  for(let previous=1;previous<=x;previous++)assert.equal(canEnter(previous,x),true);
  assert.equal(canEnter(x+1,x),false);
 }
 assert.equal(resumeLevel('20',5),1);assert.equal(resumeLevel('bad',5),1);
 assert.equal(canEnter(0,5),false);assert.equal(canEnter(2.5,5),false);
});
