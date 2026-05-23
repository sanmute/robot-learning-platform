import { useEffect, useRef, useState, useCallback } from "react";
import * as THREE from "three";
import { ShortTermMemory, STMFrame } from './components/memory/stm.js';
import { LongTermMemory, LTMPattern } from './components/memory/LTM.js';
import { ConsolidationEngine } from './components/memory/ConsolidationEngine.js';
import { DualMemoryController } from './components/memory/DualMemoryController.js';
import { ExperimentRunner, FREE_PLAY_FLAGS } from './components/ExperimentRunner.js';
import { ExperimentRunner as ExperimentRunnerV2 } from './components/ExperimentRunner_v2.js';

const N = 25;
const TRIAL_DURATION = 3600;
const OBS_R = 85, FOOD_R = 150, AGENT_R = 9, FOOD_R_PX = 7;
const SA = [0, Math.PI/4, Math.PI/2, -Math.PI/4, -Math.PI/2];

function makePat(obs, food, motor) {
  const d = Array(N).fill(-1);
  obs.forEach(i => d[i] = 1);
  food.forEach(i => d[i+5] = 1);
  if (motor==='L') d[15]=1;
  if (motor==='F') d[16]=1;
  if (motor==='R') d[17]=1;
  return d;
}

const PERFECT_PATS = [
  { name:"F→F",  data: makePat([],  [0], 'F') },
  { name:"FR→R", data: makePat([],  [1], 'R') },
  { name:"R→R",  data: makePat([],  [2], 'R') },
  { name:"FL→L", data: makePat([],  [3], 'L') },
  { name:"L→L",  data: makePat([],  [4], 'L') },
  { name:"W→R",  data: makePat([0], [],  'R') },
  { name:"WR→L", data: makePat([1], [],  'L') },
  { name:"WL→R", data: makePat([3], [],  'R') },
  { name:"○→F",  data: makePat([],  [],  'F') },
];

const PERSONALITIES = [
  { name:"AGGRESSIVE", aggression:0.85, curiosity:0.4, caution:0.3, beta:7, metab:0.8, noise:0.05 },
  { name:"CURIOUS",    aggression:0.5,  curiosity:0.9, caution:0.5, beta:4, metab:1.0, noise:0.08 },
  { name:"CAUTIOUS",   aggression:0.4,  curiosity:0.6, caution:0.9, beta:6, metab:1.2, noise:0.02 },
];

const AGENT_COLORS = [
  { hex: 0x0066ff, rgb:"0,102,255",   name:"BLUE" },
  { hex: 0x00cc77, rgb:"0,204,119",   name:"GREEN" },
  { hex: 0xff7722, rgb:"255,119,34",  name:"ORANGE" },
];

// ── SPONTANEITY ANALYZER ───────────────────────────────────────
class SpontaneityAnalyzer {
  constructor(trialData) {
    this.trialData = trialData;
  }

  measureActionEntropy(actionSequence) {
    if (!actionSequence.length) return 0;
    const counts = { 'L': 0, 'F': 0, 'R': 0 };
    actionSequence.forEach(a => counts[a]++);
    const total = actionSequence.length;
    const probs = Object.values(counts).map(c => c / total);
    return -probs.reduce((h, p) => p > 0 ? h + p * Math.log(p) : h, 0);
  }

  measureTrajectoryComplexity(positionHistory, gridSize=40) {
    if (!positionHistory.length) return 0;
    const visitedCells = new Set(
      positionHistory.map(p => 
        Math.floor(p.x/gridSize) + ',' + Math.floor(p.y/gridSize)
      )
    );
    
    let totalDistance = 0;
    for(let i=1; i<positionHistory.length; i++) {
      const dx = positionHistory[i].x - positionHistory[i-1].x;
      const dy = positionHistory[i].y - positionHistory[i-1].y;
      totalDistance += Math.sqrt(dx*dx + dy*dy);
    }
    
    const maxArea = (800/gridSize) * (600/gridSize);
    const explorationRatio = visitedCells.size / maxArea;
    const pathSmoothness = 1 / (1 + totalDistance / 100);
    
    return explorationRatio * (1 - pathSmoothness * 0.3);
  }

  measureNeuralComplexity(neuralStateHistory, attentionHistory) {
    if (!neuralStateHistory.length || !attentionHistory.length) return 0;
    
    const neuronActivations = Array(25).fill(0);
    neuralStateHistory.forEach(state => {
      state.forEach((v, i) => {
        if(v === 1) neuronActivations[i]++;
      });
    });
    
    const probs = neuronActivations.map(a => a / neuralStateHistory.length);
    const H_neural = -probs.reduce((h, p) => 
      p > 0 && p < 1 ? h + p * Math.log(p) : h, 0);
    
    const patternFreqs = Array(9).fill(0);
    attentionHistory.forEach(attn => {
      const maxIdx = attn.reduce((a, v, i) => v > attn[a] ? i : a, 0);
      patternFreqs[maxIdx]++;
    });
    
    const patternProbs = patternFreqs.map(f => f / attentionHistory.length);
    const H_attention = -patternProbs.reduce((h, p) =>
      p > 0 && p < 1 ? h + p * Math.log(p) : h, 0);
    
    return (H_neural + H_attention) / 2;
  }

  measureStrategySwitching(attentionHistory, window=60) {
    if (!attentionHistory.length) return 0;
    let switches = 0;
    let prevStrategy = 0;
    
    for(let i=0; i<attentionHistory.length; i+=window) {
      const windowAttn = attentionHistory.slice(i, Math.min(i+window, attentionHistory.length));
      if (!windowAttn.length) continue;
      
      const avgAttn = Array(9).fill(0);
      windowAttn.forEach(a => {
        a.forEach((v, j) => avgAttn[j] += v);
      });
      avgAttn.forEach((v, i) => avgAttn[i] /= windowAttn.length);
      
      const strategy = avgAttn.reduce((a, v, i) => v > avgAttn[a] ? i : a, 0);
      if(strategy !== prevStrategy) switches++;
      prevStrategy = strategy;
    }
    
    return switches;
  }

  measurePredictability(actionSequence, order=2) {
    if (actionSequence.length < order + 1) return 0.33;
    
    const markovChain = {};
    for(let i=order; i<actionSequence.length; i++) {
      const history = actionSequence.slice(i-order, i).join(',');
      const next = actionSequence[i];
      
      if(!markovChain[history]) markovChain[history] = {L:0, F:0, R:0};
      markovChain[history][next]++;
    }
    
    let totalProb = 0, count = 0;
    for(let i=order; i<actionSequence.length; i++) {
      const history = actionSequence.slice(i-order, i).join(',');
      const actual = actionSequence[i];
      
      if(markovChain[history]) {
        const total = Object.values(markovChain[history]).reduce((a,b)=>a+b);
        const prob = markovChain[history][actual] / total;
        totalProb += prob;
        count++;
      }
    }
    
    return count > 0 ? totalProb / count : 0.33;
  }

  generateReport() {
    const report = {
      metadata: this.trialData.metadata,
      agents: {}
    };
    
    for (let aid = 0; aid < 3; aid++) {
      const history = this.trialData.agents[aid];
      if (!history) continue;
      
      report.agents[aid] = {
        personality: history.personality,
        
        action_entropy: Math.round(this.measureActionEntropy(history.actions) * 100) / 100,
        trajectory_complexity: Math.round(this.measureTrajectoryComplexity(history.positions) * 100) / 100,
        neural_diversity: Math.round(this.measureNeuralComplexity(history.neuralStates, history.attentions) * 100) / 100,
        strategy_switches: this.measureStrategySwitching(history.attentions),
        predictability: Math.round(this.measurePredictability(history.actions) * 100) / 100,
        
        final_score: history.scores[history.scores.length - 1] || 0,
        total_distance: history.positions.reduce((d, p, i, arr) => {
          if (i === 0) return 0;
          const dx = p.x - arr[i-1].x;
          const dy = p.y - arr[i-1].y;
          return d + Math.sqrt(dx*dx + dy*dy);
        }, 0),
        visited_cells: new Set(
          history.positions.map(p => Math.floor(p.x/40) + ',' + Math.floor(p.y/40))
        ).size,
      };
    }
    
    return report;
  }

  downloadReport() {
    const report = this.generateReport();
    const json = JSON.stringify(report, null, 2);
    const blob = new Blob([json], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `spontaneity_${report.metadata.trial_id}.json`;
    a.click();
    URL.revokeObjectURL(url);
  }
}

// ── DATA LOGGER ────────────────────────────────────────────────
class DataLogger {
  constructor(trialId, condition, personalityIndices) {
    this.trialId = trialId;
    this.condition = condition;
    this.personalityIndices = personalityIndices;
    this.startTime = Date.now();
    this.frameData = [];
    this.agents = {};
    
    for (let i = 0; i < personalityIndices.length; i++) {
      this.agents[i] = {
        personality: PERSONALITIES[personalityIndices[i]].name,
        actions: [],
        positions: [],
        neuralStates: [],
        attentions: [],
        scores: [],
      };
    }
  }

  logFrame(frameNum, agents, agentStates) {
    for (let i = 0; i < agents.length; i++) {
      this.agents[i].actions.push(agentStates[i]?.action || 'F');
      this.agents[i].positions.push({ x: agents[i].x, y: agents[i].y });
      this.agents[i].neuralStates.push(agentStates[i]?.state || []);
      this.agents[i].attentions.push(agentStates[i]?.attn || []);
      this.agents[i].scores.push(agents[i].score);
    }
  }

  exportJSON() {
    return {
      metadata: {
        trial_id: this.trialId,
        condition: this.condition,
        personality_indices: this.personalityIndices,
        started_at: new Date(this.startTime).toISOString(),
        duration_ms: Date.now() - this.startTime,
      },
      agents: this.agents,
    };
  }
}

// ── NEURAL FUNCTIONS ───────────────────────────────────────────
function modernStep(state, patterns, beta, noiseLevel) {
  if (!patterns.length) return { newState:[...state], attn:[] };
  
  const noisyState = state.map(v => {
    if(Math.random() < noiseLevel) return Math.random() < 0.5 ? 1 : -1;
    return v;
  });
  
  const sims = patterns.map(({data}) => {
    let d=0;
    for(let i=0;i<N;i++) d+=data[i]*noisyState[i];
    return beta*d;
  });
  const mx = Math.max(...sims);
  const ex = sims.map(s => Math.exp(s-mx));
  const sm = ex.reduce((a,b)=>a+b,0);
  const attn = ex.map(e=>e/sm);
  const raw = new Float32Array(N);
  for(let k=0;k<patterns.length;k++) for(let i=0;i<N;i++) raw[i]+=patterns[k].data[i]*attn[k];
  return { newState: Array.from(raw,v=>v>=0?1:-1), attn };
}

// ── WORLD PHYSICS ──────────────────────────────────────────────
function initWorld(W, H, personalityIndices) {
  const agents = personalityIndices.map((pIdx, idx) => {
    const pers = PERSONALITIES[pIdx];
    const col = AGENT_COLORS[Math.min(idx, 2)];
    return {
      id: idx,
      color: col,
      personalityIdx: pIdx,
      personality: pers,
      x: W/2 + (idx-1)*40 + (Math.random()-0.5)*20,
      y: H/2 + (idx-1)*40 + (Math.random()-0.5)*20,
      angle: Math.random() * Math.PI * 2,
      hunger: 30,
      fatigue: 0,
      score: 0,
      visited: new Set(),
      stm: new ShortTermMemory(60),   // 60 frames, 1 second window
      ltm: new LongTermMemory(1000),  // up to 1000 consolidated patterns
      engine: null,                   // set below after stm + ltm exist
    };
  });

  // Wire ConsolidationEngine and DualMemoryController after agents are fully built
  for (const agent of agents) {
    agent.engine = new ConsolidationEngine(agent.stm, agent.ltm, {
      windowSize:        30,
      rewardThreshold:   25,
      surpriseThreshold: 0.5,
      periodicInterval:  300,
    });
    agent.controller = new DualMemoryController(agent.ltm, {
      // 0.25: fresh patterns start at reliability≈0.5 × cs≈0.4 = 0.20, so anything
      // above ~0.22 requires at least one successful strengthening pass first.
      // 0.30 (old default) was still too high — patterns were never usable early on.
      ltmConfidenceThreshold: 0.25,
      explorationRate:        0.2,
      actionWeightSTM:        0.6,
    });
  }

  return {
    agents,
    foods: Array.from({length:12}, ()=>({ x:50+Math.random()*(W-100), y:50+Math.random()*(H-100) })),
    W, H,
  };
}

function cellKey(x, y) {
  return Math.floor(x/40)+','+Math.floor(y/40);
}

function wallDist(ox, oy, angle, W, H, range) {
  const dx=Math.cos(angle), dy=Math.sin(angle);
  let t=range;
  if(dx<0) t=Math.min(t,(-ox)/dx);
  if(dx>0) t=Math.min(t,(W-ox)/dx);
  if(dy<0) t=Math.min(t,(-oy)/dy);
  if(dy>0) t=Math.min(t,(H-oy)/dy);
  return t;
}

function getSensors(agent, world, otherAgents) {
  const {x, y, angle} = agent;
  const {foods, W, H} = world;
  const obs = Array(5).fill(false), food = Array(5).fill(false), agents_nearby = Array(5).fill(false);
  
  for(let s=0;s<5;s++){
    const ra = angle + SA[s];
    const wd = wallDist(x, y, ra, W, H, OBS_R);
    obs[s] = wd < OBS_R * 0.88;
    
    for(const f of foods){
      const dx=f.x-x, dy=f.y-y, dist=Math.sqrt(dx*dx+dy*dy);
      if(dist < FOOD_R){
        let rel = Math.atan2(dy,dx) - angle;
        while(rel>Math.PI)rel-=2*Math.PI; while(rel<-Math.PI)rel+=2*Math.PI;
        if(Math.abs(rel-SA[s])<Math.PI/4) food[s]=true;
      }
    }
  }
  
  for(const other of otherAgents){
    if(other.id === agent.id) continue;
    const dx = other.x - x, dy = other.y - y, dist = Math.sqrt(dx*dx+dy*dy);
    if(dist < OBS_R){
      let rel = Math.atan2(dy, dx) - angle;
      while(rel>Math.PI)rel-=2*Math.PI; while(rel<-Math.PI)rel+=2*Math.PI;
      const idx = SA.findIndex((sa, i) => Math.abs(rel - sa) < Math.PI/4);
      if(idx >= 0) agents_nearby[idx] = true;
    }
  }
  
  return {obs, food, agents_nearby};
}

/**
 * Map the current sensor reading to one of the three LTM behavioral contexts.
 * - foraging:   food visible in any direction (active approach)
 * - avoidance:  obstacle/wall visible, no food (evasive navigation)
 * - exploration: open space, no notable stimuli (free movement)
 */
function determineContext(sensors) {
  if (sensors.food.some(f => f)) return 'foraging';
  if (sensors.obs.some(o => o))  return 'avoidance';
  return 'exploration';
}

function encodeSensors(obs, food, agents_nearby){
  const s = Array(N).fill(-1);
  for(let i=0;i<5;i++){
    if(obs[i] || agents_nearby[i]) s[i] = 1;
    if(food[i]) s[i+5] = 1;
  }
  return s;
}

function decodeMotor(state){
  const L=state[15],F=state[16],R=state[17];
  if(F===1&&L!==1&&R!==1)return'F';
  if(R===1&&L!==1)return'R';
  if(L===1)return'L';
  return 'F';
}

function stepWorld(world, agentIdx, action){
  const agent = world.agents[agentIdx];
  const {x, y, angle, hunger, fatigue} = agent;
  const {foods, W, H, agents} = world;
  const pers = agent.personality;
  
  const speedMult = Math.max(0.2, 1 - fatigue/100) * pers.aggression;
  const SPD = 2.5 * speedMult;
  const TURN = 0.068;
  
  let newAngle = angle;
  if(action==='L') newAngle -= TURN;
  if(action==='R') newAngle += TURN;
  
  let newX = x + Math.cos(newAngle) * SPD;
  let newY = y + Math.sin(newAngle) * SPD;
  
  const mg = AGENT_R + 3;
  if(newX < mg){ newX = mg; newAngle = Math.PI - newAngle; }
  if(newX > W - mg){ newX = W - mg; newAngle = Math.PI - newAngle; }
  if(newY < mg){ newY = mg; newAngle = -newAngle; }
  if(newY > H - mg){ newY = H - mg; newAngle = -newAngle; }
  
  for(const other of agents){
    if(other.id === agent.id) continue;
    const dx = other.x - newX, dy = other.y - newY;
    const dist = Math.sqrt(dx*dx + dy*dy);
    if(dist < AGENT_R * 2.2){
      if(pers.caution > 0.5){
        const ang = Math.atan2(dy, dx);
        newX = other.x - Math.cos(ang) * (AGENT_R * 2.5);
        newY = other.y - Math.sin(ang) * (AGENT_R * 2.5);
      }
    }
  }
  
  agent.x = newX;
  agent.y = newY;
  agent.angle = newAngle;
  
  agent.hunger = Math.min(100, agent.hunger + 0.4);
  agent.fatigue = Math.min(100, agent.fatigue + (action==='F' ? 0.6 * pers.metab : 0.2));
  if(action==='R' || agent.fatigue > 80) {
    agent.fatigue = Math.max(0, agent.fatigue - 0.8);
  }
  
  const cellK = cellKey(newX, newY);
  let curiosityBonus = 0;
  if(!agent.visited.has(cellK)){
    agent.visited.add(cellK);
    curiosityBonus = 1;
  }
  
  let ate = 0, reward = 0;
  for(const f of foods){
    const dx = f.x - newX, dy = f.y - newY;
    if(Math.sqrt(dx*dx + dy*dy) < AGENT_R + FOOD_R_PX + 2){
      f.x = 50 + Math.random()*(W-100);
      f.y = 50 + Math.random()*(H-100);
      ate++;
      agent.hunger = Math.max(0, agent.hunger - 40);
      agent.score += 1;
    }
  }
  
  if(ate) reward = (agent.hunger / 100) * 1.0;
  reward += curiosityBonus * 0.3 * pers.curiosity;
  
  return {ate, reward, curiosityBonus};
}

function drawWorld(ctx, world, agentStates){
  const {W, H, agents, foods} = world;
  ctx.fillStyle='#f8f9fc';
  ctx.fillRect(0, 0, W, H);
  
  ctx.strokeStyle='#d0d8e8';
  ctx.lineWidth=0.5;
  for(let gx=0; gx<W; gx+=40){
    ctx.beginPath();
    ctx.moveTo(gx, 0);
    ctx.lineTo(gx, H);
    ctx.stroke();
  }
  for(let gy=0; gy<H; gy+=40){
    ctx.beginPath();
    ctx.moveTo(0, gy);
    ctx.lineTo(W, gy);
    ctx.stroke();
  }
  
  for(const agent of agents){
    ctx.fillStyle=`rgba(${agent.color.rgb},0.06)`;
    for(const cellStr of agent.visited){
      const [cx, cy] = cellStr.split(',').map(Number);
      ctx.fillRect(cx*40, cy*40, 40, 40);
    }
  }
  
  for(const f of foods){
    const g=ctx.createRadialGradient(f.x,f.y,0,f.x,f.y,FOOD_R_PX*2.5);
    g.addColorStop(0,'rgba(0,200,100,0.9)');
    g.addColorStop(0.4,'rgba(0,140,70,0.5)');
    g.addColorStop(1,'rgba(0,60,30,0)');
    ctx.beginPath();
    ctx.arc(f.x, f.y, FOOD_R_PX*2, 0, Math.PI*2);
    ctx.fillStyle=g;
    ctx.fill();
    ctx.beginPath();
    ctx.arc(f.x, f.y, FOOD_R_PX, 0, Math.PI*2);
    ctx.fillStyle='rgba(0,220,120,0.9)';
    ctx.fill();
  }
  
  for(const agent of agents){
    const state = agentStates[agent.id];
    if(!state) continue;
    const {x, y, angle} = agent;
    const {sensors, action, dopamine} = state;
    const {obs, food, agents_nearby} = sensors;
    
    for(let s=0; s<5; s++){
      const ra = angle + SA[s];
      const dx = Math.cos(ra), dy = Math.sin(ra);
      const hasO = obs[s], hasF = food[s], hasA = agents_nearby[s];
      const rl = hasO ? wallDist(x, y, ra, 800, 600, OBS_R) : hasF ? 75 : hasA ? 70 : 35;
      ctx.beginPath();
      ctx.moveTo(x, y);
      ctx.lineTo(x + dx*rl, y + dy*rl);
      ctx.strokeStyle = hasO ? 'rgba(255,100,80,0.6)' : hasF ? 'rgba(0,180,100,0.5)' : hasA ? 'rgba(150,100,255,0.5)' : 'rgba(100,140,200,0.2)';
      ctx.lineWidth = 2;
      ctx.stroke();
    }
    
    const ac = action==='L' ? '80,150,255' : action==='R' ? '255,150,80' : agent.color.rgb;
    const ag = ctx.createRadialGradient(x, y, 0, x, y, AGENT_R*3.5);
    ag.addColorStop(0, `rgba(${ac},${0.15+dopamine*0.3})`);
    ag.addColorStop(1, 'transparent');
    ctx.beginPath();
    ctx.arc(x, y, AGENT_R*3.5, 0, Math.PI*2);
    ctx.fillStyle = ag;
    ctx.fill();
    
    ctx.beginPath();
    ctx.moveTo(x+Math.cos(angle)*AGENT_R*1.5, y+Math.sin(angle)*AGENT_R*1.5);
    ctx.lineTo(x+Math.cos(angle+2.5)*AGENT_R*0.9, y+Math.sin(angle+2.5)*AGENT_R*0.9);
    ctx.lineTo(x+Math.cos(angle-2.5)*AGENT_R*0.9, y+Math.sin(angle-2.5)*AGENT_R*0.9);
    ctx.closePath();
    ctx.fillStyle=`rgb(${ac})`;
    ctx.fill();
    ctx.strokeStyle='rgba(100,150,200,0.3)';
    ctx.lineWidth=1;
    ctx.stroke();
  }
  
  ctx.font='bold 11px "Courier New"';
  ctx.fillStyle='#1a2840';
  let yoff = 20;
  for(const agent of agents){
    ctx.fillStyle=`rgb(${agent.color.rgb})`;
    ctx.fillText(`${agent.color.name}: ${agent.score}`, 12, yoff);
    yoff += 16;
  }
}

function fibSphere(n, r){
  const phi = (1 + Math.sqrt(5)) / 2;
  return Array.from({length:n}, (_,i)=>{
    const th = Math.acos(1 - (2*i+1)/n), ps = 2*Math.PI*i/phi;
    return new THREE.Vector3(r*Math.sin(th)*Math.cos(ps), r*Math.sin(th)*Math.sin(ps), r*Math.cos(th));
  });
}

const NEURON_COLORS=[
  0xff6633,0xff6633,0xff6633,0xff6633,0xff6633,
  0x00cc77,0x00cc77,0x00cc77,0x00cc77,0x00cc77,
  0x4488ff,0x4488ff,0x4488ff,0x4488ff,0x4488ff,
  0xffaa00,0xffaa00,0xffaa00,
  0x6644ff,0x6644ff,0x6644ff,0x6644ff,0x6644ff,0x6644ff,0x6644ff,
];

// ── COMPONENT ──────────────────────────────────────────────────
export default function NeuroAgent() {
  const canvas2d = useRef(null);
  const brainDiv = useRef(null);
  const threeRef = useRef({});
  const worldState = useRef(null);
  const patsRef = useRef(PERFECT_PATS.map(p => ({...p})));
  const dopamineRef = useRef([0, 0, 0]);
  const runRef = useRef(false);
  const simAnim = useRef(null);
  const fpsRef = useRef({t:0,n:0});
  const loggerRef = useRef(null);
  const frameNumRef = useRef(0);
  const runnerRef = useRef(null);       // ExperimentRunner instance (null in free-play)

  const [running, setRunning] = useState(false);
  const [experimentMode, setExperimentMode] = useState(false);
  const [expUI, setExpUI] = useState(null); // runner.uiState snapshot for React
  const [activeAgent, setActiveAgent] = useState(0);
  const [scores, setScores] = useState([0, 0, 0]);
  const [attention, setAttention] = useState(PERFECT_PATS.map((_,i)=>i===8?1:0));
  const [topK, setTopK] = useState(8);
  const [fps, setFps] = useState(0);
  const [dopamine, setDopamine] = useState(0);
  const [hunger, setHunger] = useState(30);
  const [fatigue, setFatigue] = useState(0);
  const [agentStates, setAgentStates] = useState({0:{}, 1:{}, 2:{}});
  const [frameNum, setFrameNum] = useState(0);
  const [personalityIndices, setPersonalityIndices] = useState([0, 1, 2]);
  const [noiseOverride, setNoiseOverride] = useState(null); // 0-15%

  // ── Experiment 2 state ──
  const [selectedExp, setSelectedExp] = useState(1);
  const [complexityLevel, setComplexityLevel] = useState(1);
  const exp2RunnerRef = useRef(null);
  const [exp2Status, setExp2Status] = useState('idle'); // 'idle' | 'running' | 'complete' | 'error'
  const [exp2Progress, setExp2Progress] = useState(null);

  useEffect(()=>{
    const el = brainDiv.current;
    if(!el) return;
    const BW = el.clientWidth, BH = el.clientHeight;
    const renderer = new THREE.WebGLRenderer({antialias:true, alpha:true});
    renderer.setSize(BW, BH);
    renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    el.style.background='#ebebf0';
    el.appendChild(renderer.domElement);
    
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0xf0f2f8);
    const camera = new THREE.PerspectiveCamera(55, BW/BH, 0.1, 50);
    camera.position.z = 6;
    scene.add(new THREE.AmbientLight(0xffffff, 0.8));
    const dl = new THREE.DirectionalLight(0x6699ff, 1);
    dl.position.set(5, 5, 5);
    scene.add(dl);
    
    const root = new THREE.Group();
    scene.add(root);
    
    const positions = fibSphere(N, 2.5);
    const coreMats = positions.map(()=>new THREE.MeshStandardMaterial({color:0x4466dd, emissive:0x2255aa, emissiveIntensity:0.2, metalness:0.8, roughness:0.2}));
    const coreGeo = new THREE.SphereGeometry(0.14, 16, 16);
    positions.forEach((pos, i)=>{
      const m = new THREE.Mesh(coreGeo, coreMats[i]);
      m.position.copy(pos);
      root.add(m);
    });
    
    const glowMats = positions.map(()=>new THREE.MeshBasicMaterial({color:0x2288ff, transparent:true, opacity:0, blending:THREE.AdditiveBlending, depthWrite:false}));
    const glowGeo = new THREE.SphereGeometry(0.32, 12, 12);
    positions.forEach((pos, i)=>{
      const m = new THREE.Mesh(glowGeo, glowMats[i]);
      m.position.copy(pos);
      root.add(m);
    });
    
    const connMetas = [];
    for(let i=0; i<N; i++) for(let j=i+1; j<N; j++){
      const geo = new THREE.BufferGeometry().setFromPoints([positions[i], positions[j]]);
      const mat = new THREE.LineBasicMaterial({color:0x7799dd, transparent:true, opacity:0.06, blending:THREE.NormalBlending, depthWrite:false});
      root.add(new THREE.Line(geo, mat));
      connMetas.push({mat, i, j});
    }
    
    let autoY = 0, animId;
    const loop = ()=>{
      animId = requestAnimationFrame(loop);
      autoY += 0.004;
      root.rotation.y = autoY;
      renderer.render(scene, camera);
    };
    loop();
    
    threeRef.current = {renderer, coreMats, glowMats, connMetas};
    return ()=>{
      cancelAnimationFrame(animId);
      renderer.dispose();
      if(el.contains(renderer.domElement)) el.removeChild(renderer.domElement);
    };
  }, []);

  const updateBrain = useCallback((state, attn, dopa)=>{
    const {coreMats, glowMats, connMetas} = threeRef.current;
    if(!coreMats) return;
    
    const tk = attn.reduce((b,v,i)=>v>attn[b]?i:b, 0);
    const tp = patsRef.current[tk]?.data;
    
    state.forEach((v, i)=>{
      const on = v === 1;
      let emis = on ? NEURON_COLORS[i] : 0x8899dd;
      coreMats[i].emissive.setHex(emis);
      coreMats[i].emissiveIntensity = on ? (0.3 + dopa*0.5) : 0.15;
      coreMats[i].color.setHex(0x334488);
      glowMats[i].color.setHex(on ? NEURON_COLORS[i] : 0x8899dd);
      glowMats[i].opacity = (on ? 0.25 : 0.05) * (1 + dopa*2);
    });
    
    connMetas?.forEach(({mat, i, j})=>{
      const ss = tp && tp[i] === tp[j];
      const bo = state[i] === 1 && state[j] === 1;
      mat.color.setHex(ss ? 0x2255aa : 0xaa5588);
      mat.opacity = (bo ? (ss ? 0.35 : 0.08) : (ss ? 0.08 : 0.02)) * (1 + dopa);
    });
  }, []);

  const simLoop = useCallback((ts)=>{
    if(!runRef.current) return;
    
    const canvas = canvas2d.current;
    const ctx = canvas.getContext('2d');
    const world = worldState.current;
    const logger = loggerRef.current;
    
    const currentFrame = frameNumRef.current;
    
    if(currentFrame >= TRIAL_DURATION) {
      runRef.current = false;
      setRunning(false);
      console.log("Trial complete, analyzing...");
      if(logger) {
        const data = logger.exportJSON();
        const analyzer = new SpontaneityAnalyzer(data);
        const report = analyzer.generateReport();
        console.log("Spontaneity Report:", report);
        analyzer.downloadReport();
      }
      return;
    }
    
    const newAgentStates = {};
    const newDopamines = [...dopamineRef.current];

    // Condition flags: gates which memory subsystems are active this trial.
    // In free-play mode runnerRef is null → all systems enabled (FREE_PLAY_FLAGS).
    const flags = runnerRef.current?.isRunning
      ? runnerRef.current.conditionFlags
      : FREE_PLAY_FLAGS;

    for(let aid = 0; aid < personalityIndices.length; aid++){
      const agent = world.agents[aid];
      const pers = agent.personality;

      // Use noise override if set, otherwise use personality's default
      const effectiveNoise = noiseOverride !== null ? noiseOverride / 100 : pers.noise;

      const sensors = getSensors(agent, world, world.agents);
      const partial = encodeSensors(sensors.obs, sensors.food, sensors.agents_nearby);
      const {newState, attn} = modernStep(partial, patsRef.current, pers.beta, effectiveNoise);
      const hopfieldAction = decodeMotor(newState);
      const context = determineContext(sensors);

      // ── Action selection (condition-gated) ────────────────────────────────
      let action = hopfieldAction;
      let memSource = 'STM';
      if (flags.useController && agent.controller) {
        const result = agent.controller.selectAction(partial, hopfieldAction, context);
        action    = result.action;
        memSource = result.source;
      }

      const {reward} = stepWorld(world, aid, action);

      // ── Post-action feedback (condition-gated) ────────────────────────────
      if (flags.useController && agent.controller) {
        agent.controller.evaluateAction(partial, action, reward);
      }

      // ── STM recording (condition-gated) ──────────────────────────────────
      agent.sensoryState = partial;
      if (flags.useSTM && agent.stm) {
        const stmFrame = new STMFrame(
          currentFrame,       // timestamp
          agent.sensoryState, // [25] sensory vector
          newState,           // [25] neural activations
          action,             // 'L' | 'F' | 'R'
          reward,             // float
          attn                // [M] attention weights
        );
        agent.stm.add(stmFrame);
      }

      // ── STM → LTM consolidation (condition-gated) ─────────────────────────
      if (flags.useConsolidation && agent.engine) {
        agent.engine.update(currentFrame, currentFrame);
      }

      // ── Periodic diagnostics (every 100 frames, free-play only) ──────────
      if (!experimentMode && currentFrame % 100 === 0 && currentFrame > 0) {
        const stmStats   = agent.stm?.getStats();
        const engStats   = agent.engine?.getStats();
        const ctrlStats  = agent.controller?.stats();
        if (stmStats && engStats) {
          const elapsedSec  = (Date.now() - agent.stm.createdAt) / 1000;
          const captureRate = (stmStats.framesAdded / elapsedSec).toFixed(1);
          console.log(
            `Frame ${currentFrame} [agent ${aid}]:`,
            `STM: ${stmStats.size}/${stmStats.capacity} (${captureRate} fps,`,
            `span=${stmStats.newestTimestamp - stmStats.oldestTimestamp}f) |`,
            `LTM: ${engStats.ltmPatterns} patterns`,
            `[exp=${engStats.byContext.exploration}`,
            `for=${engStats.byContext.foraging}`,
            `avo=${engStats.byContext.avoidance}] |`,
            `Engine: +${engStats.newPatterns} new, ~${engStats.strengthened} strengthened`,
            `[R:${engStats.triggerBreakdown.reward}`,
            `S:${engStats.triggerBreakdown.surprise}`,
            `P:${engStats.triggerBreakdown.periodic}]`,
            ctrlStats
              ? `| Controller: LTM=${(ctrlStats.ltmUsageRate * 100).toFixed(1)}% conf=${ctrlStats.avgLTMConfidence.toFixed(2)}`
              : ''
          );
        }
        if (ctrlStats) {
          // Check if patterns are being USED and updated
          console.log(`Controller stats:`, ctrlStats);
        }
      }

      let dopa = newDopamines[aid];
      dopa = dopa * Math.exp(-0.016/0.5) + reward;
      newDopamines[aid] = dopa;

      const entropy = -attn.reduce((s, p) => s + (p > 0 ? p * Math.log(p) : 0), 0);
      // reward is included so ExperimentRunner.tick() can collect per-frame metrics
      newAgentStates[aid] = {sensors, action, dopamine:dopa, state:newState, attn, entropy, reward};
    }

    dopamineRef.current = newDopamines;
    drawWorld(ctx, world, newAgentStates);

    if(logger) {
      logger.logFrame(currentFrame, world.agents, newAgentStates);
    }

    // ── Experiment runner tick ────────────────────────────────────────────
    if (experimentMode && runnerRef.current?.isRunning) {
      const { trialComplete, nextCondition, isExperimentComplete }
        = runnerRef.current.tick(currentFrame, world.agents, newAgentStates);

      if (trialComplete) {
        if (isExperimentComplete) {
          runRef.current = false;
          setRunning(false);
          setExpUI({ ...runnerRef.current.uiState });
          console.log('[App] Experiment complete.');
        } else {
          // Reset world for next trial (frame counter already reset below)
          const c2 = canvas2d.current;
          worldState.current = initWorld(c2.width, c2.height, [0, 1, 2]);
          runnerRef.current.prepareAgentsForCondition(worldState.current.agents, nextCondition);
          dopamineRef.current = [0, 0, 0];
          frameNumRef.current = 0;
          setScores([0, 0, 0]);
          setExpUI({ ...runnerRef.current.uiState });
          simAnim.current = requestAnimationFrame(simLoop);
          return; // skip the normal frameNumRef++ below on the same frame
        }
      } else {
        // Keep UI in sync every 50 frames
        if (currentFrame % 50 === 0) setExpUI({ ...runnerRef.current.uiState });
      }
    }

    frameNumRef.current = currentFrame + 1;
    
    const fc = fpsRef.current;
    fc.n++;
    if(ts - fc.t > 250){
      setFps(Math.round(fc.n / ((ts - fc.t)/1000)));
      fc.t = ts;
      fc.n = 0;
      
      setFrameNum(frameNumRef.current);
      setScores(world.agents.map(a => a.score));
      
      const activeA = world.agents[activeAgent];
      const aState = newAgentStates[activeAgent];
      if(aState){
        setAttention([...aState.attn]);
        setTopK(aState.attn.reduce((b,v,i)=>v>aState.attn[b]?i:b, 0));
        setDopamine(aState.dopamine);
        updateBrain(aState.state, aState.attn, aState.dopamine);
      }
      
      setHunger(activeA.hunger);
      setFatigue(activeA.fatigue);
      setAgentStates({...newAgentStates});
    }
    
    simAnim.current = requestAnimationFrame(simLoop);
  }, [personalityIndices, activeAgent, updateBrain, noiseOverride]);

  const startTrial = useCallback((persIndices, noise=null) => {
    const c = canvas2d.current;
    const trialId = `trial_${Date.now()}_${Math.random().toString(36).slice(2, 7)}`;
    
    worldState.current = initWorld(c.width, c.height, persIndices);
    setPersonalityIndices(persIndices);
    setNoiseOverride(noise);
    
    const conditions = persIndices.map(idx => PERSONALITIES[idx].name).join("-");
    const noiseLabel = noise !== null ? `_noise${noise}` : "";
    loggerRef.current = new DataLogger(trialId + noiseLabel, conditions, persIndices);
    
    frameNumRef.current = 0;
    setFrameNum(0);
    setScores(Array(persIndices.length).fill(0));
    setHunger(30);
    setFatigue(0);
    dopamineRef.current = Array(persIndices.length).fill(0);
    
    runRef.current = true;
    setRunning(true);
    fpsRef.current = {t:0, n:0};
    simAnim.current = requestAnimationFrame(simLoop);
  }, [simLoop]);

  // ── Experiment mode ─────────────────────────────────────────────────────
  const startExperiment = useCallback(() => {
    if (running) {
      runRef.current = false;
      cancelAnimationFrame(simAnim.current);
    }

    const runner = new ExperimentRunner({ perfectPats: PERFECT_PATS, N });
    runnerRef.current = runner;

    const { condition } = runner.start();

    const c = canvas2d.current;
    worldState.current = initWorld(c.width, c.height, [0, 1, 2]);
    runner.prepareAgentsForCondition(worldState.current.agents, condition);

    setPersonalityIndices([0, 1, 2]);
    setNoiseOverride(null);
    loggerRef.current = null;  // experiment uses its own export
    frameNumRef.current = 0;
    setFrameNum(0);
    setScores([0, 0, 0]);
    setHunger(30);
    setFatigue(0);
    dopamineRef.current = [0, 0, 0];
    setExperimentMode(true);
    setExpUI({ ...runner.uiState });

    runRef.current = true;
    setRunning(true);
    fpsRef.current = {t:0, n:0};
    simAnim.current = requestAnimationFrame(simLoop);
  }, [running, simLoop]);

  const stopExperiment = useCallback(() => {
    runRef.current = false;
    cancelAnimationFrame(simAnim.current);
    setRunning(false);
    setExperimentMode(false);
    runnerRef.current = null;
    setExpUI(null);
  }, []);

  const handleRunExp2 = useCallback(async () => {
    if (exp2Status === 'running') return;
    setExp2Status('running');
    setExp2Progress({ completedTrials: 0, totalTrials: 300, percentComplete: '0.0' });
    const runner = new ExperimentRunnerV2();
    exp2RunnerRef.current = runner;
    runner.onProgressUpdate = (info) => { setExp2Progress({ ...info }); };
    try {
      await runner.runExperiment(2);
      setExp2Status(runner._stopped ? 'idle' : 'complete');
    } catch (err) {
      console.error('[App] Experiment 2 failed:', err);
      setExp2Status('error');
    } finally {
      exp2RunnerRef.current = null;
    }
  }, [exp2Status]);

  const handleStopExp2 = useCallback(() => {
    exp2RunnerRef.current?.stop();
    setExp2Status('idle');
    setExp2Progress(null);
  }, []);

  const toggleRun = useCallback(()=>{
    if(running){
      if (experimentMode) { stopExperiment(); return; }
      runRef.current = false;
      cancelAnimationFrame(simAnim.current);
      setRunning(false);
    } else {
      startTrial(personalityIndices, noiseOverride);
    }
  }, [running, experimentMode, personalityIndices, noiseOverride, startTrial, stopExperiment]);

  useEffect(()=>{
    const resize = ()=>{
      const c = canvas2d.current;
      if(!c) return;
      const p = c.parentElement;
      c.width = p.clientWidth;
      c.height = p.clientHeight;
      worldState.current = initWorld(c.width, c.height, personalityIndices);
      const ctx = c.getContext('2d');
      ctx.fillStyle = '#f8f9fc';
      ctx.fillRect(0, 0, c.width, c.height);
    };
    resize();
    window.addEventListener('resize', resize);
    return ()=>window.removeEventListener('resize', resize);
  }, [personalityIndices]);

  return(
    <div style={{display:"flex", height:"100vh", background:"#f5f7fa", color:"#1a2840", fontFamily:"'Courier New',monospace", overflow:"hidden"}}>

      <div style={{width:320, flexShrink:0, display:"flex", flexDirection:"column", gap:11, padding:"16px 14px", background:"#e8ecf8", borderRight:"1px solid #d0d8e8", overflowY:"auto"}}>
        <div>
          <div style={{fontSize:8, letterSpacing:3, color:"#1a2880", marginBottom:3}}>SPONTANEITY LAB</div>
          <div style={{fontSize:16, fontWeight:700, letterSpacing:2, color:"#0066ff", lineHeight:1.2}}>
            NEURO<span style={{color:"#ff7722"}}>SPON</span>
            <span style={{fontSize:7, color:"#2a4880", marginLeft:8, fontWeight:400}}>v0.9</span>
          </div>
          <div style={{fontSize:7, color:"#2a4880", marginTop:2, letterSpacing:1}}>EMERGENCE ANALYSIS</div>
        </div>

        <div>
          <div style={{fontSize:6, letterSpacing:2, color:"#0066ff", marginBottom:3, fontWeight:700}}>NOISE OVERRIDE</div>
          <div style={{display:"flex", alignItems:"center", gap:8}}>
            <input 
              type="range" 
              min={0} 
              max={15} 
              step={1} 
              value={noiseOverride ?? 5}
              onChange={(e) => setNoiseOverride(+e.target.value)}
              disabled={running}
              style={{flex:1, accentColor:"#0066ff", cursor:running?"default":"pointer"}}
            />
            <div style={{fontSize:9, fontWeight:700, color:"#ff7722", width:28}}>
              {noiseOverride ?? "—"}%
            </div>
          </div>
          <div style={{fontSize:5, color:"#2a4880", marginTop:4}}>
            (Override personality noise with fixed %)
          </div>
        </div>

        <div>
          <div style={{fontSize:6, letterSpacing:2, color:"#0066ff", marginBottom:3, fontWeight:700}}>PRESETS</div>
          <button onClick={()=>startTrial([0,1,2], null)} disabled={running} style={{width:"100%", padding:"6px", fontSize:7, marginBottom:4, background:"#ffffff", border:"1px solid #a0a8c8", borderRadius:3, cursor:running?"default":"pointer"}}>
            Heterogeneous (default noise)
          </button>
          <button onClick={()=>startTrial([0,1,2], 0)} disabled={running} style={{width:"100%", padding:"6px", fontSize:7, marginBottom:4, background:"#ffffff", border:"1px solid #a0a8c8", borderRadius:3, cursor:running?"default":"pointer"}}>
            Test: 0% Noise (deterministic)
          </button>
          <button onClick={()=>startTrial([0,1,2], 5)} disabled={running} style={{width:"100%", padding:"6px", fontSize:7, marginBottom:4, background:"#ffffff", border:"1px solid #a0a8c8", borderRadius:3, cursor:running?"default":"pointer"}}>
            Test: 5% Noise (optimal?)
          </button>
          <button onClick={()=>startTrial([0,1,2], 10)} disabled={running} style={{width:"100%", padding:"6px", fontSize:7, background:"#ffffff", border:"1px solid #a0a8c8", borderRadius:3, cursor:running?"default":"pointer"}}>
            Test: 10% Noise (chaotic)
          </button>
        </div>

        {/* ── Mode toggle ── */}
        <div style={{display:"flex", gap:4}}>
          <button
            onClick={()=>{ if(!running){ setExperimentMode(false); runnerRef.current=null; setExpUI(null); } }}
            disabled={running}
            style={{flex:1, padding:"5px 2px", fontSize:6, letterSpacing:1, fontFamily:"inherit",
              background: !experimentMode ? "#0066ff" : "#ffffff",
              color:      !experimentMode ? "#ffffff"  : "#0066ff",
              border:"1px solid #0066ff", borderRadius:3, cursor:running?"default":"pointer"}}
          >FREE PLAY</button>
          <button
            onClick={()=>{ if(!running){ setExperimentMode(true); } }}
            disabled={running}
            style={{flex:1, padding:"5px 2px", fontSize:6, letterSpacing:1, fontFamily:"inherit",
              background: experimentMode ? "#aa00cc" : "#ffffff",
              color:      experimentMode ? "#ffffff"  : "#aa00cc",
              border:"1px solid #aa00cc", borderRadius:3, cursor:running?"default":"pointer"}}
          >EXPERIMENT</button>
        </div>

        {/* ── Free-play controls ── */}
        {!experimentMode && (
          <div style={{display:"flex", gap:6}}>
            <button onClick={toggleRun} style={{flex:2, padding:"9px 4px", fontSize:8, letterSpacing:2, fontFamily:"inherit", background:running?"#ff5533":"#00aa44", color:"#ffffff", border:"1px solid #0066ff", borderRadius:3, cursor:"pointer"}}>
              {running?"■ STOP":"▶ START"}
            </button>
          </div>
        )}

        {/* ── Experiment sub-selector (EXP 1 / EXP 2) ── */}
        {experimentMode && (
          <div style={{display:"flex", gap:3}}>
            <button
              onClick={()=>setSelectedExp(1)}
              disabled={running || exp2Status==='running'}
              style={{flex:1, padding:"4px 2px", fontSize:6, letterSpacing:1, fontFamily:"inherit",
                background: selectedExp===1 ? "#aa00cc" : "#ffffff",
                color:      selectedExp===1 ? "#ffffff"  : "#aa00cc",
                border:"1px solid #aa00cc", borderRadius:3, cursor:"pointer"}}
            >EXP 1</button>
            <button
              onClick={()=>setSelectedExp(2)}
              disabled={running || exp2Status==='running'}
              style={{flex:1, padding:"4px 2px", fontSize:6, letterSpacing:1, fontFamily:"inherit",
                background: selectedExp===2 ? "#006699" : "#ffffff",
                color:      selectedExp===2 ? "#ffffff"  : "#006699",
                border:"1px solid #006699", borderRadius:3, cursor:"pointer"}}
            >EXP 2</button>
          </div>
        )}

        {/* ── Experiment 1 controls ── */}
        {experimentMode && selectedExp===1 && (
          <div>
            {!running ? (
              <button onClick={startExperiment} style={{width:"100%", padding:"9px 4px", fontSize:8, letterSpacing:2, fontFamily:"inherit", background:"#aa00cc", color:"#ffffff", border:"none", borderRadius:3, cursor:"pointer"}}>
                ▶ RUN EXPERIMENT (20 trials)
              </button>
            ) : (
              <button onClick={stopExperiment} style={{width:"100%", padding:"9px 4px", fontSize:8, letterSpacing:2, fontFamily:"inherit", background:"#ff5533", color:"#ffffff", border:"none", borderRadius:3, cursor:"pointer"}}>
                ■ STOP EXPERIMENT
              </button>
            )}
          </div>
        )}

        {/* ── Experiment 2 controls ── */}
        {experimentMode && selectedExp===2 && (
          <div>
            <div style={{fontSize:6, color:"#335577", marginBottom:4, lineHeight:1.6}}>
              <strong>Env Complexity Scaling</strong><br/>
              5 levels × 4 conditions × 5 trials × 3 agents = 300 runs<br/>
              L1: 5 obs / L3: 30 obs / L5: 100 obs
            </div>
            {exp2Status !== 'running' ? (
              <button onClick={handleRunExp2} style={{width:"100%", padding:"9px 4px", fontSize:8, letterSpacing:2, fontFamily:"inherit", background:"#006699", color:"#ffffff", border:"none", borderRadius:3, cursor:"pointer"}}>
                ▶ RUN EXP 2 (300 runs)
              </button>
            ) : (
              <button onClick={handleStopExp2} style={{width:"100%", padding:"9px 4px", fontSize:8, letterSpacing:2, fontFamily:"inherit", background:"#ff5533", color:"#ffffff", border:"none", borderRadius:3, cursor:"pointer"}}>
                ■ ABORT EXP 2
              </button>
            )}
          </div>
        )}

        {/* ── Experiment 2 progress panel ── */}
        {experimentMode && selectedExp===2 && exp2Progress && (
          <div style={{fontSize:6, padding:"8px", background:"#e8f4ff", borderRadius:3, border:"1px solid #88bbdd"}}>
            <div style={{color:"#006699", fontWeight:700, marginBottom:4}}>
              {exp2Status==='complete' ? '✓ EXP 2 COMPLETE — JSON saved' : `EXP 2 RUNNING…`}
            </div>
            <div style={{marginBottom:2}}>
              Trials: <span style={{float:"right", fontWeight:700}}>{exp2Progress.completedTrials ?? 0}/{exp2Progress.totalTrials ?? 300}</span>
            </div>
            <div style={{height:5, background:"#cce0f0", borderRadius:2, marginBottom:3}}>
              <div style={{height:"100%", borderRadius:2, background:"#006699",
                width:`${exp2Progress.percentComplete ?? 0}%`, transition:"width 0.4s"}}/>
            </div>
            <div style={{marginBottom:2}}>
              Level: <span style={{float:"right", fontWeight:700}}>{exp2Progress.currentLevel ?? '—'}/5</span>
            </div>
            <div style={{marginBottom:2}}>
              Condition: <span style={{float:"right", fontWeight:700}}>{exp2Progress.currentCondition ?? '—'}</span>
            </div>
            <div>
              Complete: <span style={{float:"right", fontWeight:700}}>{exp2Progress.percentComplete ?? '0.0'}%</span>
            </div>
          </div>
        )}

        {/* ── Experiment 2 idle hint ── */}
        {experimentMode && selectedExp===2 && !exp2Progress && (
          <div style={{fontSize:6, padding:"8px", background:"#f0f8ff", borderRadius:3, border:"1px solid #b0d0e8", color:"#335577"}}>
            Runs 5 complexity levels (L1–L5) under 4 memory conditions (A/B/C/D). Results auto-download as JSON on completion.
          </div>
        )}

        {/* ── Experiment 1 live status ── */}
        {experimentMode && selectedExp===1 && expUI ? (
          <div style={{fontSize:6, padding:"8px", background:"#f5eeff", borderRadius:3, border:"1px solid #cc88ff"}}>
            <div style={{color:"#aa00cc", fontWeight:700, marginBottom:4}}>
              {expUI.phase === 'complete' ? '✓ EXPERIMENT COMPLETE' : `TRIAL ${expUI.currentTrial}/${expUI.totalTrials}`}
            </div>
            <div style={{marginBottom:2}}>Condition: <span style={{float:"right", fontWeight:700, color:"#aa00cc"}}>{expUI.conditionLabel}</span></div>
            <div style={{marginBottom:2, color:"#665599", fontSize:5}}>{expUI.conditionDesc}</div>

            {/* Overall progress bar */}
            <div style={{height:4, background:"#ddd", borderRadius:2, marginBottom:3}}>
              <div style={{height:"100%", borderRadius:2, background:"#aa00cc", width:`${expUI.progress*100}%`, transition:"width 0.3s"}}/>
            </div>
            <div style={{marginBottom:2}}>Overall: <span style={{float:"right"}}>{expUI.completedCount}/{expUI.totalTrials} done</span></div>

            {/* Trial progress bar */}
            <div style={{height:3, background:"#ddd", borderRadius:2, marginBottom:3}}>
              <div style={{height:"100%", borderRadius:2, background:"#6600cc", width:`${expUI.trialProgress*100}%`}}/>
            </div>
            <div style={{marginBottom:4}}>Trial frame: <span style={{float:"right"}}>{expUI.trialFrame}/600</span></div>

            {/* Per-condition counts */}
            <div style={{display:"flex", gap:3}}>
              {['A','B','C','D'].map(c=>(
                <div key={c} style={{flex:1, textAlign:"center", padding:"2px", background: expUI.condition===c ? "#cc88ff" : "#eee", borderRadius:2, fontSize:6}}>
                  <div style={{fontWeight:700}}>{c}</div>
                  <div>{expUI.completedByCondition?.[c] ?? 0}/5</div>
                </div>
              ))}
            </div>
          </div>
        ) : experimentMode && selectedExp===1 ? (
          <div style={{fontSize:6, padding:"8px", background:"#ffffff", borderRadius:3, border:"1px solid #d0d8e8"}}>
            <div>Frame: <span style={{float:"right", fontWeight:700}}>{frameNum}/{TRIAL_DURATION}</span></div>
            <div>Progress: <span style={{float:"right", fontWeight:700}}>{Math.round(frameNum/TRIAL_DURATION*100)}%</span></div>
            <div>FPS: <span style={{float:"right", fontWeight:700}}>{fps}</span></div>
            <div style={{marginTop:4, borderTop:"1px solid #d0d8e8", paddingTop:4}}>Noise: <span style={{float:"right", fontWeight:700}}>{noiseOverride !== null ? noiseOverride+"%" : "default"}</span></div>
          </div>
        ) : !experimentMode ? (
          <div style={{fontSize:6, padding:"8px", background:"#ffffff", borderRadius:3, border:"1px solid #d0d8e8"}}>
            <div>Frame: <span style={{float:"right", fontWeight:700}}>{frameNum}/{TRIAL_DURATION}</span></div>
            <div>Progress: <span style={{float:"right", fontWeight:700}}>{Math.round(frameNum/TRIAL_DURATION*100)}%</span></div>
            <div>FPS: <span style={{float:"right", fontWeight:700}}>{fps}</span></div>
            <div style={{marginTop:4, borderTop:"1px solid #d0d8e8", paddingTop:4}}>Noise: <span style={{float:"right", fontWeight:700}}>{noiseOverride !== null ? noiseOverride+"%" : "default"}</span></div>
          </div>
        ) : null}

        <div>
          <div style={{fontSize:6, letterSpacing:2, color:"#0066ff", marginBottom:3, fontWeight:700}}>SCORES</div>
          {AGENT_COLORS.slice(0, personalityIndices.length).map((col, idx)=>(
            <div key={idx} style={{display:"flex", justifyContent:"space-between", marginBottom:2, padding:"4px 6px", background:"#ffffff", borderRadius:3}}>
              <div style={{fontSize:7, color:`rgb(${col.rgb})`}}>{PERSONALITIES[personalityIndices[idx]].name}</div>
              <div style={{fontSize:11, fontWeight:700, color:`rgb(${col.rgb})`}}>{scores[idx]}</div>
            </div>
          ))}
        </div>

        <div style={{fontSize:6, lineHeight:2.2, color:"#2a4880", padding:"8px", background:"#ffffff", borderRadius:3, border:"1px solid #d0d8e8", marginTop:"auto"}}>
          {experimentMode && selectedExp===2
            ? <><span style={{color:"#006699"}}>▶</span> 300 agent-runs, async<br/><span style={{color:"#006699"}}>↓</span> Auto-downloads JSON<br/><span style={{color:"#006699"}}>✓</span> Check downloads folder</>
            : experimentMode
            ? <><span style={{color:"#aa00cc"}}>▶</span> 20 trials, random order<br/><span style={{color:"#aa00cc"}}>↓</span> JSON per trial + summary<br/><span style={{color:"#aa00cc"}}>✓</span> Check downloads folder</>
            : <><span style={{color:"#0066ff"}}>✓</span> Metrics logged<br/><span style={{color:"#0066ff"}}>✓</span> Analyze at end<br/><span style={{color:"#0066ff"}}>✓</span> Check downloads!</>
          }
        </div>
      </div>

      <div style={{flex:1, display:"flex", flexDirection:"column", overflow:"hidden"}}>
        <div style={{flex:1, position:"relative", overflow:"hidden"}}>
          <canvas ref={canvas2d} style={{display:"block", width:"100%", height:"100%"}} />
          {!running&&(
            <div style={{position:"absolute", top:"50%", left:"50%", transform:"translate(-50%,-50%)", textAlign:"center", pointerEvents:"none"}}>
              <div style={{fontSize:14, letterSpacing:4, color:"#2a4880", marginBottom:8, fontWeight:700}}>READY</div>
              <div style={{fontSize:9, color:"#4a6880", letterSpacing:2}}>Set noise & start trial</div>
            </div>
          )}
        </div>
        <div ref={brainDiv} style={{height:180, flexShrink:0, borderTop:"1px solid #d0d8e8", position:"relative"}} />
      </div>

      <style>{`* { box-sizing:border-box; } button:hover:not(:disabled) { filter:brightness(0.95); }`}</style>
    </div>
  );
}