import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import {gunzipSync} from 'node:zlib';
import {createRequire} from 'node:module';
import {decodeReplay} from '../replay-codec.mjs';
import {prepareRecording} from '../campaign-client.mjs';
const require=createRequire(import.meta.url),R=require('../race.js');
const folder=new URL('../local-results/campaign/',import.meta.url);
test('All campaign courses preserve recorded anatomy/activity and correct race clocks',{skip:!fs.existsSync(new URL('manifest.json',folder))},()=>{
 const catalogue=JSON.parse(fs.readFileSync(new URL('manifest.json',folder))),base=JSON.parse(fs.readFileSync(new URL('../model/model.json',import.meta.url)));
 assert.equal(catalogue.accepted,100);
 for(const entry of catalogue.levels){
  const record=decodeReplay(gunzipSync(fs.readFileSync(new URL(entry.file,folder)))),result=prepareRecording(record,entry,base),d=result.description;
  assert.equal(result.frames,record.frames);assert.ok(record.frames.at(-1).success);
  assert.equal(d.brain_coordinates_3d,base.description.brain_coordinates_3d);
  assert.equal(d.campaign.level,entry.id);assert.equal(d.colliders.length,entry.walls*2);
  assert.ok(Math.abs(result.frames.at(-1).tick*d.dt-entry.flyRaceSeconds/d.fly_speed_factor)<1e-6);
  assert.ok(d.player_dock_angle>=3.5*Math.PI/180&&d.player_dock_angle<=30*Math.PI/180);
  assert.equal(R.create(d).time,0);
  assert.equal(record.frames[1].packets.length,record.header.sender.length*record.header.sweeps);
 }
});
test('Reject recordings with wrong identity or no successful terminal state',()=>{
 assert.throws(()=>prepareRecording({header:{level:2},frames:[]},{id:1},{}));
});
