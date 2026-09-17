// The race rate is a contract, never a function of model throughput.
export const RACE_SPEED=3.30;
export const TICK_SECONDS=.08;
export const TICK_BUDGET_MS=TICK_SECONDS*1000/RACE_SPEED;
export const LOOKAHEAD_TICKS=128;
export function canStream(msPerTick){return Number.isFinite(msPerTick)&&msPerTick>0&&msPerTick<=TICK_BUDGET_MS*.65;}
export function raceDuration(wallSeconds){return Math.max(0,wallSeconds)*RACE_SPEED;}
