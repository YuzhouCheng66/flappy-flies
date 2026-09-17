import test from 'node:test';
import assert from 'node:assert/strict';
import {COURSE_IDS,COURSE_COUNT,sourceCourse,previewLevel} from '../campaign-selection.mjs';
test('30 unique audited recordings; only two single-wall tutorials',()=>{
 assert.equal(COURSE_COUNT,30);assert.equal(new Set(COURSE_IDS).size,30);
 assert.equal(COURSE_IDS.filter(x=>x<=20).length,2);
 assert.equal(sourceCourse(3),21);assert.equal(sourceCourse(30),100);
 assert.throws(()=>sourceCourse(31));assert.throws(()=>sourceCourse(0));
});
test('Independent hard-course preview is local and bounded',()=>{
 assert.equal(previewLevel('?try=30','127.0.0.1'),30);
 assert.equal(previewLevel('?try=30','yuzhoucheng.com'),null);
 for(const s of ['', '?try=31','?try=-1','?try=1.5'])assert.equal(previewLevel(s,'127.0.0.1'),null);
});
