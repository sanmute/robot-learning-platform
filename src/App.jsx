import { useEffect, useRef, useState, useCallback } from "react";
import * as THREE from "three";
import { ShortTermMemory, STMFrame } from './components/memory/stm.js';
import { LongTermMemory, LTMPattern } from './components/memory/LTM.js';
import { ConsolidationEngine } from './components/memory/ConsolidationEngine.js';
import { DualMemoryController } from './components/memory/DualMemoryController.js';
import { ExperimentRunner, FREE_PLAY_FLAGS } from './components/ExperimentRunner.js';
import { ExperimentRunner as ExperimentRunnerV2 } from './components/ExperimentRunner_v2.js';
import { SIMULATION_CONFIG, EXPERIMENT_CONFIG as EXP_CFG } from './components/EXPERIMENT_CONFIG.js';

const N            = SIMULATION_CONFIG.HOPFIELD_NEURONS;
const TRIAL_DURATION = 3600;   // free-play UI timer; experiments use EXP_CFG.TRIAL_DURATION_FRAMES
const OBS_R        = SIMULATION_CONFIG.OBSTACLE_DETECTION_RANGE;
const FOOD_R       = SIMULATION_CONFIG.FOOD_DETECTION_RANGE;
const AGENT_R      = SIMULATION_CONFIG.AGENT_RADIUS;
const FOOD_R_PX    = SIMULATION_CONFIG.FOOD_RADIUS;
const SA           = SIMULATION_CONFIG.SENSOR_ANGLES;

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
      ltmConfidenceThreshold: 0.01,
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

  // ── Experiment 2 & 3 state ──
  const [selectedExp, setSelectedExp] = useState(1);
  const [complexityLevel, setComplexityLevel] = useState(1);
  const exp2RunnerRef = useRef(null);
  const [exp2Status, setExp2Status] = useState('idle'); // 'idle' | 'running' | 'complete' | 'error'
  const [exp2Progress, setExp2Progress] = useState(null);
  const exp3RunnerRef = useRef(null);
  const [exp3Status, setExp3Status] = useState('idle');
  const [exp3Progress, setExp3Progress] = useState(null);
  const exp4RunnerRef = useRef(null);
  const [exp4Status, setExp4Status] = useState('idle');
  const [exp4Progress, setExp4Progress] = useState(null);
  const exp45RunnerRef = useRef(null);
  const [exp45Status, setExp45Status] = useState('idle');
  const [exp45Progress, setExp45Progress] = useState(null);
  const exp5RunnerRef = useRef(null);
  const [exp5Status, setExp5Status] = useState('idle');
  const [exp5Progress, setExp5Progress] = useState(null);
  const exp55RunnerRef = useRef(null);
  const [exp55Status, setExp55Status] = useState('idle');
  const [exp55Progress, setExp55Progress] = useState(null);
  const exp555RunnerRef = useRef(null);
  const [exp555Status, setExp555Status] = useState('idle');
  const [exp555Progress, setExp555Progress] = useState(null);
  const exp6RunnerRef = useRef(null);
  const [exp6Status, setExp6Status] = useState('idle');
  const [exp6Progress, setExp6Progress] = useState(null);
  const exp8RunnerRef = useRef(null);
  const [exp8Status, setExp8Status] = useState('idle');
  const [exp8Progress, setExp8Progress] = useState(null);
  const exp9RunnerRef = useRef(null);
  const [exp9Status, setExp9Status] = useState('idle');
  const [exp9Progress, setExp9Progress] = useState(null);
  const exp10RunnerRef = useRef(null);
  const [exp10Status, setExp10Status] = useState('idle');
  const [exp10Progress, setExp10Progress] = useState(null);
  const exp10bRunnerRef = useRef(null);
  const [exp10bStatus, setExp10bStatus] = useState('idle');
  const [exp10bProgress, setExp10bProgress] = useState(null);

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

  const handleRunExp3 = useCallback(async () => {
    if (exp3Status === 'running' || exp2Status === 'running') return;
    setExp3Status('running');
    // 3 profiles × 5 levels × 4 conditions × 15 trials = 900
    setExp3Progress({ completedTrials: 0, totalTrials: 900, percentComplete: '0.0' });
    const runner = new ExperimentRunnerV2();
    exp3RunnerRef.current = runner;
    runner.onProgressUpdate = (info) => { setExp3Progress({ ...info }); };
    try {
      await runner.runExperiment(3);
      setExp3Status(runner._stopped ? 'idle' : 'complete');
    } catch (err) {
      console.error('[App] Experiment 3 failed:', err);
      setExp3Status('error');
    } finally {
      exp3RunnerRef.current = null;
    }
  }, [exp3Status, exp2Status]);

  const handleStopExp3 = useCallback(() => {
    exp3RunnerRef.current?.stop();
    setExp3Status('idle');
    setExp3Progress(null);
  }, []);

  const handleRunExp4 = useCallback(async () => {
    if (exp4Status === 'running' || exp3Status === 'running' || exp2Status === 'running') return;
    setExp4Status('running');
    // Phase 1 scaling: 4×2×4×5=160  |  Phase 2 interference: 3×2×2×5=60  → 220 total
    setExp4Progress({ completedTrials: 0, totalTrials: 220, percentComplete: '0.0' });
    const runner = new ExperimentRunnerV2();
    exp4RunnerRef.current = runner;
    runner.onProgressUpdate = (info) => { setExp4Progress({ ...info }); };
    try {
      await runner.runExperiment(4);
      setExp4Status(runner._stopped ? 'idle' : 'complete');
    } catch (err) {
      console.error('[App] Experiment 4 failed:', err);
      setExp4Status('error');
    } finally {
      exp4RunnerRef.current = null;
    }
  }, [exp4Status, exp3Status, exp2Status]);

  const handleStopExp4 = useCallback(() => {
    exp4RunnerRef.current?.stop();
    setExp4Status('idle');
    setExp4Progress(null);
  }, []);

  const handleRunExp45 = useCallback(async () => {
    if (exp45Status === 'running' || exp4Status === 'running' || exp3Status === 'running' || exp2Status === 'running') return;
    setExp45Status('running');
    // 2 variants × 4 counts × 2 levels × 2 conditions × 5 trials = 160
    setExp45Progress({ completedTrials: 0, totalTrials: 160, percentComplete: '0.0' });
    const runner = new ExperimentRunnerV2();
    exp45RunnerRef.current = runner;
    runner.onProgressUpdate = (info) => { setExp45Progress({ ...info }); };
    try {
      await runner.runExperiment(4.5);
      setExp45Status(runner._stopped ? 'idle' : 'complete');
    } catch (err) {
      console.error('[App] Experiment 4.5 failed:', err);
      setExp45Status('error');
    } finally {
      exp45RunnerRef.current = null;
    }
  }, [exp45Status, exp4Status, exp3Status, exp2Status]);

  const handleStopExp45 = useCallback(() => {
    exp45RunnerRef.current?.stop();
    setExp45Status('idle');
    setExp45Progress(null);
  }, []);

  const handleRunExp5 = useCallback(async () => {
    if (exp5Status === 'running' || exp45Status === 'running' || exp4Status === 'running'
        || exp3Status === 'running' || exp2Status === 'running') return;
    setExp5Status('running');
    // 5 variants × 2 conditions × 5 trials = 50
    setExp5Progress({ completedTrials: 0, totalTrials: 50, percentComplete: '0.0' });
    const runner = new ExperimentRunnerV2();
    exp5RunnerRef.current = runner;
    runner.onProgressUpdate = (info) => { setExp5Progress({ ...info }); };
    try {
      await runner.runExperiment(5);
      setExp5Status(runner._stopped ? 'idle' : 'complete');
    } catch (err) {
      console.error('[App] Experiment 5 failed:', err);
      setExp5Status('error');
    } finally {
      exp5RunnerRef.current = null;
    }
  }, [exp5Status, exp45Status, exp4Status, exp3Status, exp2Status]);

  const handleStopExp5 = useCallback(() => {
    exp5RunnerRef.current?.stop();
    setExp5Status('idle');
    setExp5Progress(null);
  }, []);

  const handleRunExp55 = useCallback(async () => {
    if (exp55Status === 'running' || exp5Status === 'running' || exp45Status === 'running'
        || exp4Status === 'running' || exp3Status === 'running' || exp2Status === 'running') return;
    setExp55Status('running');
    // 2 × 10 training + 2 × 5 obj × 2 cond × 5 testing = 120
    setExp55Progress({ completedTrials: 0, totalTrials: 120, percentComplete: '0.0',
                       phase: 'training', trainedPatterns: 0 });
    const runner = new ExperimentRunnerV2();
    exp55RunnerRef.current = runner;
    runner.onProgressUpdate = (info) => { setExp55Progress({ ...info }); };
    try {
      await runner.runExperiment(5.5);
      setExp55Status(runner._stopped ? 'idle' : 'complete');
    } catch (err) {
      console.error('[App] Experiment 5.5 failed:', err);
      setExp55Status('error');
    } finally {
      exp55RunnerRef.current = null;
    }
  }, [exp55Status, exp5Status, exp45Status, exp4Status, exp3Status, exp2Status]);

  const handleStopExp55 = useCallback(() => {
    exp55RunnerRef.current?.stop();
    setExp55Status('idle');
    setExp55Progress(null);
  }, []);

  const handleRunExp555 = useCallback(async () => {
    if (exp555Status === 'running' || exp55Status === 'running' || exp5Status === 'running'
        || exp45Status === 'running' || exp4Status === 'running'
        || exp3Status === 'running'  || exp2Status === 'running') return;
    setExp555Status('running');
    // 5 × 10 training + 5 × 5 obj × 2 cond × 5 testing = 300
    setExp555Progress({ completedTrials: 0, totalTrials: 300, percentComplete: '0.0',
                        phase: 'training', trainedPatterns: 0 });
    const runner = new ExperimentRunnerV2();
    exp555RunnerRef.current = runner;
    runner.onProgressUpdate = (info) => { setExp555Progress({ ...info }); };
    try {
      await runner.runExperiment('5.5.5');
      setExp555Status(runner._stopped ? 'idle' : 'complete');
    } catch (err) {
      console.error('[App] Experiment 5.5.5 failed:', err);
      setExp555Status('error');
    } finally {
      exp555RunnerRef.current = null;
    }
  }, [exp555Status, exp55Status, exp5Status, exp45Status, exp4Status, exp3Status, exp2Status]);

  const handleStopExp555 = useCallback(() => {
    exp555RunnerRef.current?.stop();
    setExp555Status('idle');
    setExp555Progress(null);
  }, []);

  const handleRunExp6 = useCallback(async () => {
    if (exp6Status === 'running' || exp555Status === 'running' || exp55Status === 'running'
      || exp5Status === 'running' || exp45Status === 'running' || exp4Status === 'running'
      || exp3Status === 'running' || exp2Status === 'running') return;
    setExp6Status('running');
    setExp6Progress(null);
    const runner = new ExperimentRunnerV2();
    runner.onProgressUpdate = (info) => setExp6Progress(info);
    exp6RunnerRef.current = runner;
    try {
      await runner.runExperiment(6);
      setExp6Status('complete');
    } finally {
      exp6RunnerRef.current = null;
    }
  }, [exp6Status, exp555Status, exp55Status, exp5Status, exp45Status, exp4Status, exp3Status, exp2Status]);

  const handleStopExp6 = useCallback(() => {
    exp6RunnerRef.current?.stop();
    setExp6Status('idle');
    setExp6Progress(null);
  }, []);

  const handleRunExp8 = useCallback(async () => {
    if (exp8Status === 'running' || exp6Status === 'running' || exp555Status === 'running'
      || exp55Status === 'running' || exp5Status === 'running' || exp45Status === 'running'
      || exp4Status === 'running' || exp3Status === 'running' || exp2Status === 'running') return;
    setExp8Status('running');
    setExp8Progress(null);
    const runner = new ExperimentRunnerV2();
    runner.onProgressUpdate = (info) => setExp8Progress(info);
    exp8RunnerRef.current = runner;
    try {
      await runner.runExperiment(8);
      setExp8Status('complete');
    } finally {
      exp8RunnerRef.current = null;
    }
  }, [exp8Status, exp6Status, exp555Status, exp55Status, exp5Status, exp45Status, exp4Status, exp3Status, exp2Status]);

  const handleStopExp8 = useCallback(() => {
    exp8RunnerRef.current?.stop();
    setExp8Status('idle');
    setExp8Progress(null);
  }, []);

  const handleRunExp9 = useCallback(async () => {
    if (exp9Status === 'running' || exp8Status === 'running' || exp6Status === 'running'
      || exp555Status === 'running' || exp55Status === 'running' || exp5Status === 'running'
      || exp45Status === 'running' || exp4Status === 'running'
      || exp3Status === 'running' || exp2Status === 'running') return;
    setExp9Status('running');
    setExp9Progress(null);
    const runner = new ExperimentRunnerV2();
    runner.onProgressUpdate = (info) => setExp9Progress(info);
    exp9RunnerRef.current = runner;
    try {
      await runner.runExperiment(9);
      setExp9Status('complete');
    } finally {
      exp9RunnerRef.current = null;
    }
  }, [exp9Status, exp8Status, exp6Status, exp555Status, exp55Status, exp5Status, exp45Status, exp4Status, exp3Status, exp2Status]);

  const handleStopExp9 = useCallback(() => {
    exp9RunnerRef.current?.stop();
    setExp9Status('idle');
    setExp9Progress(null);
  }, []);

  const handleRunExp10 = useCallback(async () => {
    if (exp10Status === 'running' || exp9Status === 'running' || exp8Status === 'running'
      || exp6Status === 'running' || exp555Status === 'running' || exp55Status === 'running'
      || exp5Status === 'running' || exp45Status === 'running' || exp4Status === 'running'
      || exp3Status === 'running' || exp2Status === 'running') return;
    setExp10Status('running');
    setExp10Progress(null);
    const runner = new ExperimentRunnerV2();
    runner.onProgressUpdate = (info) => setExp10Progress(info);
    exp10RunnerRef.current = runner;
    try {
      await runner.runExperiment(10);
      setExp10Status('complete');
    } finally {
      exp10RunnerRef.current = null;
    }
  }, [exp10Status, exp9Status, exp8Status, exp6Status, exp555Status, exp55Status, exp5Status, exp45Status, exp4Status, exp3Status, exp2Status]);

  const handleStopExp10 = useCallback(() => {
    exp10RunnerRef.current?.stop();
    setExp10Status('idle');
    setExp10Progress(null);
  }, []);

  const handleRunExp10b = useCallback(async () => {
    if (exp10bStatus === 'running' || exp10Status === 'running' || exp9Status === 'running'
      || exp8Status === 'running' || exp6Status === 'running' || exp555Status === 'running'
      || exp55Status === 'running' || exp5Status === 'running' || exp45Status === 'running'
      || exp4Status === 'running' || exp3Status === 'running' || exp2Status === 'running') return;
    setExp10bStatus('running');
    setExp10bProgress(null);
    const runner = new ExperimentRunnerV2();
    runner.onProgressUpdate = (info) => setExp10bProgress(info);
    exp10bRunnerRef.current = runner;
    try {
      await runner.runExperiment('10b');
      setExp10bStatus('complete');
    } finally {
      exp10bRunnerRef.current = null;
    }
  }, [exp10bStatus, exp10Status, exp9Status, exp8Status, exp6Status, exp555Status, exp55Status, exp5Status, exp45Status, exp4Status, exp3Status, exp2Status]);

  const handleStopExp10b = useCallback(() => {
    exp10bRunnerRef.current?.stop();
    setExp10bStatus('idle');
    setExp10bProgress(null);
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

        {/* ── Experiment sub-selector (two rows) ── */}
        {experimentMode && (() => {
          const anyRunning = running || exp2Status==='running' || exp3Status==='running' || exp4Status==='running' || exp45Status==='running' || exp5Status==='running' || exp55Status==='running' || exp555Status==='running' || exp6Status==='running' || exp8Status==='running' || exp9Status==='running' || exp10Status==='running' || exp10bStatus==='running';
          const btn = (n, color, label) => (
            <button key={n}
              onClick={()=>setSelectedExp(n)}
              disabled={anyRunning}
              style={{flex:1, padding:"4px 1px", fontSize:5.5, letterSpacing:0.5, fontFamily:"inherit",
                background: selectedExp===n ? color : "#ffffff",
                color:      selectedExp===n ? "#ffffff" : color,
                border:`1px solid ${color}`, borderRadius:3, cursor:"pointer"}}
            >{label}</button>
          );
          return (
            <div style={{display:"flex", flexDirection:"column", gap:3}}>
              <div style={{display:"flex", gap:3}}>
                {btn(1, "#aa00cc", "EXP 1")}
                {btn(2, "#006699", "EXP 2")}
                {btn(3, "#cc6600", "EXP 3")}
              </div>
              <div style={{display:"flex", gap:3}}>
                {btn(4,   "#006644", "EXP 4")}
                {btn(4.5, "#007755", "EXP 4.5")}
                {btn(5,   "#884400", "EXP 5")}
              </div>
              <div style={{display:"flex", gap:3}}>
                {btn(5.5,   "#5500aa", "EXP 5.5")}
                {btn('5.5.5', "#770077", "EXP 5.5.5")}
              </div>
              <div style={{display:"flex", gap:3}}>
                {btn(6, "#004488", "EXP 6")}
                {btn(8, "#224400", "EXP 8")}
                {btn(9, "#005566", "EXP 9")}
              </div>
              <div style={{display:"flex", gap:3}}>
                {btn(10,   "#330044", "EXP 10a")}
                {btn('10b',"#550022", "EXP 10b")}
              </div>
            </div>
          );
        })()}

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

        {/* ── Experiment 3 controls ── */}
        {experimentMode && selectedExp===3 && (
          <div>
            <div style={{fontSize:6, color:"#553311", marginBottom:4, lineHeight:1.6}}>
              <strong>Robustness: Real-World &amp; Space</strong><br/>
              3 profiles × 5 levels × 4 cond × 15 trials = 900 runs<br/>
              Warehouse · Physics · Space stressors
            </div>
            {exp3Status !== 'running' ? (
              <button onClick={handleRunExp3} style={{width:"100%", padding:"9px 4px", fontSize:8, letterSpacing:2, fontFamily:"inherit", background:"#cc6600", color:"#ffffff", border:"none", borderRadius:3, cursor:"pointer"}}>
                ▶ RUN EXP 3 (900 runs)
              </button>
            ) : (
              <button onClick={handleStopExp3} style={{width:"100%", padding:"9px 4px", fontSize:8, letterSpacing:2, fontFamily:"inherit", background:"#ff5533", color:"#ffffff", border:"none", borderRadius:3, cursor:"pointer"}}>
                ■ ABORT EXP 3
              </button>
            )}
          </div>
        )}

        {/* ── Experiment 3 progress panel ── */}
        {experimentMode && selectedExp===3 && exp3Progress && (
          <div style={{fontSize:6, padding:"8px", background:"#fff3e8", borderRadius:3, border:"1px solid #ddaa77"}}>
            <div style={{color:"#cc6600", fontWeight:700, marginBottom:4}}>
              {exp3Status==='complete' ? '✓ EXP 3 COMPLETE — JSON saved' : 'EXP 3 RUNNING…'}
            </div>
            <div style={{marginBottom:2}}>
              Trials: <span style={{float:"right", fontWeight:700}}>{exp3Progress.completedTrials ?? 0}/{exp3Progress.totalTrials ?? 900}</span>
            </div>
            <div style={{height:5, background:"#f0d8b0", borderRadius:2, marginBottom:3}}>
              <div style={{height:"100%", borderRadius:2, background:"#cc6600",
                width:`${exp3Progress.percentComplete ?? 0}%`, transition:"width 0.4s"}}/>
            </div>
            {/* Profile badges */}
            <div style={{display:"flex", gap:3, marginBottom:3}}>
              {['warehouse','physics','space'].map(p => (
                <div key={p} style={{flex:1, textAlign:"center", padding:"2px",
                  background: exp3Progress.currentProfile===p ? "#cc6600" : "#f0d8b0",
                  color:      exp3Progress.currentProfile===p ? "#fff"    : "#885500",
                  borderRadius:2, fontSize:5, fontWeight:700}}>
                  {p === 'warehouse' ? '🏭' : p === 'physics' ? '🌍' : '🚀'}
                </div>
              ))}
            </div>
            <div style={{marginBottom:2}}>
              Level: <span style={{float:"right", fontWeight:700}}>{exp3Progress.currentLevel ?? '—'}/4</span>
            </div>
            <div style={{marginBottom:2}}>
              Stressor: <span style={{float:"right", fontWeight:700, maxWidth:"55%", overflow:"hidden", textOverflow:"ellipsis", whiteSpace:"nowrap"}}>{exp3Progress.stressorLabel ?? '—'}</span>
            </div>
            <div style={{marginBottom:2}}>
              Condition: <span style={{float:"right", fontWeight:700}}>{exp3Progress.currentCondition ?? '—'}</span>
            </div>
            <div>
              Complete: <span style={{float:"right", fontWeight:700}}>{exp3Progress.percentComplete ?? '0.0'}%</span>
            </div>
          </div>
        )}

        {/* ── Experiment 3 idle hint ── */}
        {experimentMode && selectedExp===3 && !exp3Progress && (
          <div style={{fontSize:6, padding:"8px", background:"#fff8f0", borderRadius:3, border:"1px solid #e0c090", color:"#553311"}}>
            <strong>Warehouse</strong> — sensor noise (σ 0→50%)<br/>
            <strong>Physics</strong> — gravity (1g→0g)<br/>
            <strong>Space</strong> — radiation + drift + gravity<br/>
            Results auto-download as JSON.
          </div>
        )}

        {/* ── Experiment 4 controls ── */}
        {experimentMode && selectedExp===4 && (
          <div>
            <div style={{fontSize:6, color:"#224433", marginBottom:4, lineHeight:1.6}}>
              <strong>Multi-Agent Coordination</strong><br/>
              📈 Scaling: 4 counts × 2 levels × 4 cond × 5 = 160<br/>
              🔀 Interference: 3 envs × 2 levels × 2 cond × 5 = 60<br/>
              Total: <strong>220 trials</strong>
            </div>
            {exp4Status !== 'running' ? (
              <button onClick={handleRunExp4} style={{width:"100%", padding:"9px 4px", fontSize:8, letterSpacing:2, fontFamily:"inherit", background:"#006644", color:"#ffffff", border:"none", borderRadius:3, cursor:"pointer"}}>
                ▶ RUN EXP 4 (220 runs)
              </button>
            ) : (
              <button onClick={handleStopExp4} style={{width:"100%", padding:"9px 4px", fontSize:8, letterSpacing:2, fontFamily:"inherit", background:"#ff5533", color:"#ffffff", border:"none", borderRadius:3, cursor:"pointer"}}>
                ■ ABORT EXP 4
              </button>
            )}
          </div>
        )}

        {/* ── Experiment 4 progress panel ── */}
        {experimentMode && selectedExp===4 && exp4Progress && (
          <div style={{fontSize:6, padding:"8px", background:"#e8fff4", borderRadius:3, border:"1px solid #77dd99"}}>
            <div style={{color:"#006644", fontWeight:700, marginBottom:4}}>
              {exp4Status==='complete' ? '✓ EXP 4 COMPLETE — JSON saved' : 'EXP 4 RUNNING…'}
            </div>
            <div style={{marginBottom:2}}>
              Trials: <span style={{float:"right", fontWeight:700}}>{exp4Progress.completedTrials ?? 0}/{exp4Progress.totalTrials ?? 220}</span>
            </div>
            <div style={{height:5, background:"#b0eece", borderRadius:2, marginBottom:3}}>
              <div style={{height:"100%", borderRadius:2, background:"#006644",
                width:`${exp4Progress.percentComplete ?? 0}%`, transition:"width 0.4s"}}/>
            </div>
            {/* Phase badges */}
            <div style={{display:"flex", gap:3, marginBottom:3}}>
              {['scaling','interference'].map(p => (
                <div key={p} style={{flex:1, textAlign:"center", padding:"2px",
                  background: exp4Progress.testType===p ? "#006644" : "#b0eece",
                  color:      exp4Progress.testType===p ? "#fff"    : "#224433",
                  borderRadius:2, fontSize:5, fontWeight:700}}>
                  {p === 'scaling' ? '📈 Scaling' : '🔀 Interference'}
                </div>
              ))}
            </div>
            {exp4Progress.testType === 'scaling' && (
              <div style={{marginBottom:2}}>
                Agents: <span style={{float:"right", fontWeight:700}}>{exp4Progress.currentAgentCount ?? '—'}</span>
              </div>
            )}
            {exp4Progress.testType === 'interference' && (
              <div style={{marginBottom:2}}>
                Env: <span style={{float:"right", fontWeight:700, maxWidth:"55%", overflow:"hidden", textOverflow:"ellipsis", whiteSpace:"nowrap"}}>{exp4Progress.currentEnvLabel ?? '—'}</span>
              </div>
            )}
            <div style={{marginBottom:2}}>
              Level: <span style={{float:"right", fontWeight:700}}>{exp4Progress.currentLevel ?? '—'}</span>
            </div>
            <div style={{marginBottom:2}}>
              Condition: <span style={{float:"right", fontWeight:700}}>{exp4Progress.currentCondition ?? '—'}</span>
            </div>
            <div>
              Complete: <span style={{float:"right", fontWeight:700}}>{exp4Progress.percentComplete ?? '0.0'}%</span>
            </div>
          </div>
        )}

        {/* ── Experiment 4 idle hint ── */}
        {experimentMode && selectedExp===4 && !exp4Progress && (
          <div style={{fontSize:6, padding:"8px", background:"#f0fff8", borderRadius:3, border:"1px solid #a0ddbc", color:"#224433"}}>
            <strong>Scaling</strong> — 1, 2, 3, 5 agents · L2 &amp; L3<br/>
            <strong>Interference</strong> — 400×300 · 800×600 · 1200×900<br/>
            Tests per-agent perf degradation &amp; collision impact.<br/>
            Results auto-download as JSON.
          </div>
        )}

        {/* ── Experiment 4.5 controls ── */}
        {experimentMode && selectedExp===4.5 && (
          <div>
            <div style={{fontSize:6, color:"#1a3328", marginBottom:4, lineHeight:1.6}}>
              <strong>Shared LTM Consolidation</strong><br/>
              🔒 Independent vs 🔗 Shared LTM pool<br/>
              2 variants × 4 counts × 2 levels × 2 cond × 5 = <strong>160 trials</strong><br/>
              Does sharing restore learning at scale?
            </div>
            {exp45Status !== 'running' ? (
              <button onClick={handleRunExp45} style={{width:"100%", padding:"9px 4px", fontSize:8, letterSpacing:2, fontFamily:"inherit", background:"#007755", color:"#ffffff", border:"none", borderRadius:3, cursor:"pointer"}}>
                ▶ RUN EXP 4.5 (160 runs)
              </button>
            ) : (
              <button onClick={handleStopExp45} style={{width:"100%", padding:"9px 4px", fontSize:8, letterSpacing:2, fontFamily:"inherit", background:"#ff5533", color:"#ffffff", border:"none", borderRadius:3, cursor:"pointer"}}>
                ■ ABORT EXP 4.5
              </button>
            )}
          </div>
        )}

        {/* ── Experiment 4.5 progress panel ── */}
        {experimentMode && selectedExp===4.5 && exp45Progress && (
          <div style={{fontSize:6, padding:"8px", background:"#e8fff4", borderRadius:3, border:"1px solid #55cc99"}}>
            <div style={{color:"#007755", fontWeight:700, marginBottom:4}}>
              {exp45Status==='complete' ? '✓ EXP 4.5 COMPLETE — JSON saved' : 'EXP 4.5 RUNNING…'}
            </div>
            <div style={{marginBottom:2}}>
              Trials: <span style={{float:"right", fontWeight:700}}>{exp45Progress.completedTrials ?? 0}/{exp45Progress.totalTrials ?? 160}</span>
            </div>
            <div style={{height:5, background:"#aaeedd", borderRadius:2, marginBottom:3}}>
              <div style={{height:"100%", borderRadius:2, background:"#007755",
                width:`${exp45Progress.percentComplete ?? 0}%`, transition:"width 0.4s"}}/>
            </div>
            {/* Variant badges */}
            <div style={{display:"flex", gap:3, marginBottom:3}}>
              {['independent','shared'].map(v => (
                <div key={v} style={{flex:1, textAlign:"center", padding:"2px",
                  background: exp45Progress.currentVariant===v ? "#007755" : "#aaeedd",
                  color:      exp45Progress.currentVariant===v ? "#fff"    : "#1a3328",
                  borderRadius:2, fontSize:5, fontWeight:700}}>
                  {v === 'independent' ? '🔒 Indep.' : '🔗 Shared'}
                </div>
              ))}
            </div>
            <div style={{marginBottom:2}}>
              Agents: <span style={{float:"right", fontWeight:700}}>{exp45Progress.currentAgentCount ?? '—'}</span>
            </div>
            <div style={{marginBottom:2}}>
              Level: <span style={{float:"right", fontWeight:700}}>{exp45Progress.currentLevel ?? '—'}</span>
            </div>
            <div style={{marginBottom:2}}>
              Condition: <span style={{float:"right", fontWeight:700}}>{exp45Progress.currentCondition ?? '—'}</span>
            </div>
            {exp45Progress.sharedPoolSize > 0 && (
              <div style={{marginBottom:2}}>
                Pool patterns: <span style={{float:"right", fontWeight:700, color:"#007755"}}>{exp45Progress.sharedPoolSize}</span>
              </div>
            )}
            <div>
              Complete: <span style={{float:"right", fontWeight:700}}>{exp45Progress.percentComplete ?? '0.0'}%</span>
            </div>
          </div>
        )}

        {/* ── Experiment 4.5 idle hint ── */}
        {experimentMode && selectedExp===4.5 && !exp45Progress && (
          <div style={{fontSize:6, padding:"8px", background:"#f0fff8", borderRadius:3, border:"1px solid #88ddbb", color:"#1a3328"}}>
            <strong>🔒 Independent</strong> — Exp 4 baseline, private LTMs<br/>
            <strong>🔗 Shared</strong> — all agents read/write one LTM pool<br/>
            Same scaling grid: 1, 2, 3, 5 agents × L2/L3.<br/>
            Tests if shared memory restores D vs A advantage.<br/>
            Results auto-download as JSON.
          </div>
        )}

        {/* ── Experiment 5 controls ── */}
        {experimentMode && selectedExp===5 && (
          <div>
            <div style={{fontSize:6, color:"#553311", marginBottom:4, lineHeight:1.6}}>
              <strong>Reward Structure Variation</strong><br/>
              5 variants × 2 cond (A/D) × 5 trials = <strong>50 runs</strong><br/>
              📦🔋🛡️⚡⚖️ Do memories generalise?
            </div>
            {exp5Status !== 'running' ? (
              <button onClick={handleRunExp5} style={{width:"100%", padding:"9px 4px", fontSize:8, letterSpacing:2, fontFamily:"inherit", background:"#884400", color:"#ffffff", border:"none", borderRadius:3, cursor:"pointer"}}>
                ▶ RUN EXP 5 (50 runs)
              </button>
            ) : (
              <button onClick={handleStopExp5} style={{width:"100%", padding:"9px 4px", fontSize:8, letterSpacing:2, fontFamily:"inherit", background:"#ff5533", color:"#ffffff", border:"none", borderRadius:3, cursor:"pointer"}}>
                ■ ABORT EXP 5
              </button>
            )}
          </div>
        )}

        {/* ── Experiment 5 progress panel ── */}
        {experimentMode && selectedExp===5 && exp5Progress && (
          <div style={{fontSize:6, padding:"8px", background:"#fff4e8", borderRadius:3, border:"1px solid #ddaa66"}}>
            <div style={{color:"#884400", fontWeight:700, marginBottom:4}}>
              {exp5Status==='complete' ? '✓ EXP 5 COMPLETE — JSON saved' : 'EXP 5 RUNNING…'}
            </div>
            <div style={{marginBottom:2}}>
              Trials: <span style={{float:"right", fontWeight:700}}>{exp5Progress.completedTrials ?? 0}/{exp5Progress.totalTrials ?? 50}</span>
            </div>
            <div style={{height:5, background:"#f0d0a0", borderRadius:2, marginBottom:3}}>
              <div style={{height:"100%", borderRadius:2, background:"#884400",
                width:`${exp5Progress.percentComplete ?? 0}%`, transition:"width 0.4s"}}/>
            </div>
            {/* Variant badges */}
            <div style={{display:"flex", gap:2, marginBottom:3, flexWrap:"wrap"}}>
              {[
                {name:'baseline',   emoji:'📦'},
                {name:'efficiency', emoji:'🔋'},
                {name:'accuracy',   emoji:'🛡️'},
                {name:'speed',      emoji:'⚡'},
                {name:'balance',    emoji:'⚖️'},
              ].map(v => (
                <div key={v.name} style={{flex:"0 0 auto", padding:"2px 4px",
                  background: exp5Progress.currentVariant===v.name ? "#884400" : "#f0d0a0",
                  color:      exp5Progress.currentVariant===v.name ? "#fff"    : "#553311",
                  borderRadius:2, fontSize:5, fontWeight:700}}>
                  {v.emoji}
                </div>
              ))}
            </div>
            <div style={{marginBottom:2}}>
              Variant: <span style={{float:"right", fontWeight:700, maxWidth:"55%", overflow:"hidden", textOverflow:"ellipsis", whiteSpace:"nowrap"}}>{exp5Progress.variantLabel ?? '—'}</span>
            </div>
            <div style={{marginBottom:2}}>
              Condition: <span style={{float:"right", fontWeight:700}}>{exp5Progress.currentCondition ?? '—'}</span>
            </div>
            <div>
              Complete: <span style={{float:"right", fontWeight:700}}>{exp5Progress.percentComplete ?? '0.0'}%</span>
            </div>
          </div>
        )}

        {/* ── Experiment 5 idle hint ── */}
        {experimentMode && selectedExp===5 && !exp5Progress && (
          <div style={{fontSize:6, padding:"8px", background:"#fff8f0", borderRadius:3, border:"1px solid #e0c090", color:"#553311"}}>
            <strong>📦 Baseline</strong> — maximise food collected<br/>
            <strong>🔋 Efficiency</strong> — food per energy spent<br/>
            <strong>🛡️ Accuracy</strong> — minimal wall contact<br/>
            <strong>⚡ Speed</strong> — time-pressured double reward<br/>
            <strong>⚖️ Balance</strong> — multi-objective combined<br/>
            Compares A (no memory) vs D (full dual).<br/>
            Results auto-download as JSON.
          </div>
        )}

        {/* ── Experiment 5.5 controls ── */}
        {experimentMode && selectedExp===5.5 && (
          <div>
            <div style={{fontSize:6, color:"#330066", marginBottom:4, lineHeight:1.6}}>
              <strong>Multi-Objective Learning</strong><br/>
              🎯 Train on ALL 5 rewards simultaneously<br/>
              Phase 1: 2 × 10 training trials → shared LTM<br/>
              Phase 2: 5 obj × 2 cond × 5 = 50 test trials/variant<br/>
              Total: <strong>120 trials</strong> · avg vs weighted
            </div>
            {exp55Status !== 'running' ? (
              <button onClick={handleRunExp55} style={{width:"100%", padding:"9px 4px", fontSize:8, letterSpacing:2, fontFamily:"inherit", background:"#5500aa", color:"#ffffff", border:"none", borderRadius:3, cursor:"pointer"}}>
                ▶ RUN EXP 5.5 (120 runs)
              </button>
            ) : (
              <button onClick={handleStopExp55} style={{width:"100%", padding:"9px 4px", fontSize:8, letterSpacing:2, fontFamily:"inherit", background:"#ff5533", color:"#ffffff", border:"none", borderRadius:3, cursor:"pointer"}}>
                ■ ABORT EXP 5.5
              </button>
            )}
          </div>
        )}

        {/* ── Experiment 5.5 progress panel ── */}
        {experimentMode && selectedExp===5.5 && exp55Progress && (
          <div style={{fontSize:6, padding:"8px", background:"#f3e8ff", borderRadius:3, border:"1px solid #bb88ee"}}>
            <div style={{color:"#5500aa", fontWeight:700, marginBottom:4}}>
              {exp55Status==='complete' ? '✓ EXP 5.5 COMPLETE — JSON saved' : 'EXP 5.5 RUNNING…'}
            </div>
            <div style={{marginBottom:2}}>
              Trials: <span style={{float:"right", fontWeight:700}}>{exp55Progress.completedTrials ?? 0}/{exp55Progress.totalTrials ?? 120}</span>
            </div>
            <div style={{height:5, background:"#ddbcff", borderRadius:2, marginBottom:3}}>
              <div style={{height:"100%", borderRadius:2, background:"#5500aa",
                width:`${exp55Progress.percentComplete ?? 0}%`, transition:"width 0.4s"}}/>
            </div>
            {/* Phase badges */}
            <div style={{display:"flex", gap:3, marginBottom:3}}>
              {['training','testing'].map(p => (
                <div key={p} style={{flex:1, textAlign:"center", padding:"2px",
                  background: exp55Progress.phase===p ? "#5500aa" : "#ddbcff",
                  color:      exp55Progress.phase===p ? "#fff"    : "#330066",
                  borderRadius:2, fontSize:5, fontWeight:700}}>
                  {p === 'training' ? '🏋️ Training' : '🧪 Testing'}
                </div>
              ))}
            </div>
            {/* Variant badges */}
            <div style={{display:"flex", gap:3, marginBottom:3}}>
              {['average','weighted'].map(v => (
                <div key={v} style={{flex:1, textAlign:"center", padding:"2px",
                  background: exp55Progress.currentVariant===v ? "#5500aa" : "#ddbcff",
                  color:      exp55Progress.currentVariant===v ? "#fff"    : "#330066",
                  borderRadius:2, fontSize:5, fontWeight:700}}>
                  {v === 'average' ? '⚖️ Avg' : '🎯 Wtd'}
                </div>
              ))}
            </div>
            {exp55Progress.phase === 'training' && (
              <>
                <div style={{marginBottom:2}}>
                  Train trial: <span style={{float:"right", fontWeight:700}}>{exp55Progress.trainTrial ?? '—'}/{exp55Progress.trainingTrials ?? 10}</span>
                </div>
                <div style={{marginBottom:2}}>
                  Patterns so far: <span style={{float:"right", fontWeight:700, color:"#5500aa"}}>{exp55Progress.trainedPatterns ?? 0}</span>
                </div>
              </>
            )}
            {exp55Progress.phase === 'testing' && (
              <>
                <div style={{marginBottom:2}}>
                  Objective: <span style={{float:"right", fontWeight:700}}>{exp55Progress.currentObjective ?? '—'}</span>
                </div>
                <div style={{marginBottom:2}}>
                  Condition: <span style={{float:"right", fontWeight:700}}>{exp55Progress.currentCondition ?? '—'}</span>
                </div>
                <div style={{marginBottom:2}}>
                  Trained patterns: <span style={{float:"right", fontWeight:700, color:"#5500aa"}}>{exp55Progress.totalTrainedPatterns ?? 0}</span>
                </div>
              </>
            )}
            <div>
              Complete: <span style={{float:"right", fontWeight:700}}>{exp55Progress.percentComplete ?? '0.0'}%</span>
            </div>
          </div>
        )}

        {/* ── Experiment 5.5 idle hint ── */}
        {experimentMode && selectedExp===5.5 && !exp55Progress && (
          <div style={{fontSize:6, padding:"8px", background:"#f8f0ff", borderRadius:3, border:"1px solid #cc99ff", color:"#330066"}}>
            <strong>⚖️ Average</strong> — all 5 rewards equal weight<br/>
            <strong>🎯 Weighted</strong> — commercial priority mix<br/>
            Phase 1 accumulates patterns into one shared LTM<br/>
            across 10 training trials per variant.<br/>
            Phase 2 copies trained LTM → tests each objective.<br/>
            Key metric: gen. index vs Exp 5 baseline.<br/>
            Results auto-download as JSON.
          </div>
        )}

        {/* ── Experiment 5.5.5 controls ── */}
        {experimentMode && selectedExp==='5.5.5' && (
          <div>
            <div style={{fontSize:6, color:"#440033", marginBottom:4, lineHeight:1.6}}>
              <strong>Weight Optimisation</strong><br/>
              🔍 Grid search: 5 weight combos<br/>
              Same pipeline as Exp 5.5, directly comparable<br/>
              5 × (10 train + 50 test) = <strong>300 trials</strong>
            </div>
            {exp555Status !== 'running' ? (
              <button onClick={handleRunExp555} style={{width:"100%", padding:"9px 4px", fontSize:8, letterSpacing:2, fontFamily:"inherit", background:"#770077", color:"#ffffff", border:"none", borderRadius:3, cursor:"pointer"}}>
                ▶ RUN EXP 5.5.5 (300 runs)
              </button>
            ) : (
              <button onClick={handleStopExp555} style={{width:"100%", padding:"9px 4px", fontSize:8, letterSpacing:2, fontFamily:"inherit", background:"#ff5533", color:"#ffffff", border:"none", borderRadius:3, cursor:"pointer"}}>
                ■ ABORT EXP 5.5.5
              </button>
            )}
          </div>
        )}

        {/* ── Experiment 5.5.5 progress panel ── */}
        {experimentMode && selectedExp==='5.5.5' && exp555Progress && (
          <div style={{fontSize:6, padding:"8px", background:"#ffe8ff", borderRadius:3, border:"1px solid #cc77cc"}}>
            <div style={{color:"#770077", fontWeight:700, marginBottom:4}}>
              {exp555Status==='complete' ? '✓ EXP 5.5.5 COMPLETE — JSON saved' : 'EXP 5.5.5 RUNNING…'}
            </div>
            <div style={{marginBottom:2}}>
              Trials: <span style={{float:"right", fontWeight:700}}>{exp555Progress.completedTrials ?? 0}/{exp555Progress.totalTrials ?? 300}</span>
            </div>
            <div style={{height:5, background:"#eebcee", borderRadius:2, marginBottom:3}}>
              <div style={{height:"100%", borderRadius:2, background:"#770077",
                width:`${exp555Progress.percentComplete ?? 0}%`, transition:"width 0.4s"}}/>
            </div>
            {/* Phase badges */}
            <div style={{display:"flex", gap:3, marginBottom:3}}>
              {['training','testing'].map(p => (
                <div key={p} style={{flex:1, textAlign:"center", padding:"2px",
                  background: exp555Progress.phase===p ? "#770077" : "#eebcee",
                  color:      exp555Progress.phase===p ? "#fff"    : "#440033",
                  borderRadius:2, fontSize:5, fontWeight:700}}>
                  {p === 'training' ? '🏋️ Training' : '🧪 Testing'}
                </div>
              ))}
            </div>
            {/* Combo mini-badges */}
            <div style={{display:"flex", gap:2, marginBottom:3, flexWrap:"wrap"}}>
              {['current','pro_food','balanced','efficiency_first','smart_balance'].map(c => (
                <div key={c} style={{flex:"0 0 auto", padding:"1px 3px",
                  background: exp555Progress.currentCombo===c ? "#770077" : "#eebcee",
                  color:      exp555Progress.currentCombo===c ? "#fff"    : "#440033",
                  borderRadius:2, fontSize:4.5, fontWeight:700}}>
                  {c === 'current'          ? '🎯cur'
                   : c === 'pro_food'       ? '🍎pfood'
                   : c === 'balanced'       ? '⚖️bal'
                   : c === 'efficiency_first' ? '🔋eff'
                   : '🌟smart'}
                </div>
              ))}
            </div>
            {exp555Progress.phase === 'training' ? (
              <>
                <div style={{marginBottom:2}}>
                  Combo: <span style={{float:"right", fontWeight:700, maxWidth:"55%", overflow:"hidden", textOverflow:"ellipsis", whiteSpace:"nowrap"}}>{exp555Progress.comboLabel ?? '—'}</span>
                </div>
                <div style={{marginBottom:2}}>
                  Train trial: <span style={{float:"right", fontWeight:700}}>{exp555Progress.trainTrial ?? '—'}/{exp555Progress.trainingTrials ?? 10}</span>
                </div>
                <div style={{marginBottom:2}}>
                  Patterns: <span style={{float:"right", fontWeight:700, color:"#770077"}}>{exp555Progress.trainedPatterns ?? 0}</span>
                </div>
              </>
            ) : (
              <>
                <div style={{marginBottom:2}}>
                  Combo: <span style={{float:"right", fontWeight:700, maxWidth:"50%", overflow:"hidden", textOverflow:"ellipsis", whiteSpace:"nowrap"}}>{exp555Progress.comboLabel ?? '—'}</span>
                </div>
                <div style={{marginBottom:2}}>
                  Objective: <span style={{float:"right", fontWeight:700}}>{exp555Progress.currentObjective ?? '—'}</span>
                </div>
                <div style={{marginBottom:2}}>
                  Condition: <span style={{float:"right", fontWeight:700}}>{exp555Progress.currentCondition ?? '—'}</span>
                </div>
              </>
            )}
            <div>
              Complete: <span style={{float:"right", fontWeight:700}}>{exp555Progress.percentComplete ?? '0.0'}%</span>
            </div>
          </div>
        )}

        {/* ── Experiment 5.5.5 idle hint ── */}
        {experimentMode && selectedExp==='5.5.5' && !exp555Progress && (
          <div style={{fontSize:6, padding:"8px", background:"#fff0ff", borderRadius:3, border:"1px solid #ddaadd", color:"#440033"}}>
            <strong>🎯 Current</strong> — Exp 5.5 weighted (control)<br/>
            <strong>🍎 Pro-Food</strong> — baseline:0.40 (recovery)<br/>
            <strong>⚖️ Balanced</strong> — equal 0.20 all (Exp 5.5 avg)<br/>
            <strong>🔋 Eff-First</strong> — efficiency:0.30 (upper bound)<br/>
            <strong>🌟 Smart</strong> — hypothesis winner<br/>
            Ranked by mean D-vs-A gen. index.<br/>
            JSON includes _ranking + _winner.<br/>
            Results auto-download as JSON.
          </div>
        )}

        {/* ── Experiment 6 controls ── */}
        {experimentMode && selectedExp===6 && (
          <div>
            <div style={{fontSize:6, color:"#002244", marginBottom:4, lineHeight:1.6}}>
              <strong>Transfer Learning</strong><br/>
              10 training + 5 domains × 2 cond × 5 trials = <strong>60 trials</strong><br/>
              🏭🚀📡⚡🛡️ Do patterns transfer?
            </div>
            {exp6Status !== 'running' ? (
              <button onClick={handleRunExp6} style={{width:"100%", padding:"9px 4px", fontSize:8, letterSpacing:2, fontFamily:"inherit", background:"#004488", color:"#ffffff", border:"none", borderRadius:3, cursor:"pointer"}}>
                ▶ RUN EXP 6 (60 trials)
              </button>
            ) : (
              <button onClick={handleStopExp6} style={{width:"100%", padding:"9px 4px", fontSize:8, letterSpacing:2, fontFamily:"inherit", background:"#ff5533", color:"#ffffff", border:"none", borderRadius:3, cursor:"pointer"}}>
                ■ ABORT EXP 6
              </button>
            )}
          </div>
        )}

        {/* ── Experiment 6 progress panel ── */}
        {experimentMode && selectedExp===6 && exp6Progress && (
          <div style={{fontSize:6, padding:"8px", background:"#e8f0ff", borderRadius:3, border:"1px solid #6699cc"}}>
            <div style={{color:"#004488", fontWeight:700, marginBottom:4}}>
              {exp6Status==='complete' ? '✓ EXP 6 COMPLETE — JSON saved' : 'EXP 6 RUNNING…'}
            </div>
            <div style={{marginBottom:2}}>
              Trials: <span style={{float:"right", fontWeight:700}}>{exp6Progress.completedTrials ?? 0}/{exp6Progress.totalTrials ?? 60}</span>
            </div>
            <div style={{height:5, background:"#b8cce8", borderRadius:2, marginBottom:3}}>
              <div style={{height:"100%", borderRadius:2, background:"#004488",
                width:`${exp6Progress.percentComplete ?? 0}%`, transition:"width 0.4s"}}/>
            </div>
            {/* Phase badges */}
            <div style={{display:"flex", gap:2, marginBottom:3}}>
              {['training','transfer'].map(p => (
                <div key={p} style={{flex:1, textAlign:"center", padding:"2px 3px",
                  background: exp6Progress.phase===p ? "#004488" : "#b8cce8",
                  color:      exp6Progress.phase===p ? "#fff"    : "#002244",
                  borderRadius:2, fontSize:5, fontWeight:700}}>
                  {p==='training' ? '⚙️ TRAIN' : '🚀 TRANSFER'}
                </div>
              ))}
            </div>
            {/* Domain badges */}
            <div style={{display:"flex", gap:2, marginBottom:3, flexWrap:"wrap"}}>
              {[
                {name:'warehouse', emoji:'🏭'},
                {name:'physics',   emoji:'🚀'},
                {name:'noise',     emoji:'📡'},
                {name:'speed',     emoji:'⚡'},
                {name:'safety',    emoji:'🛡️'},
              ].map(d => (
                <div key={d.name} style={{flex:"0 0 auto", padding:"2px 4px",
                  background: exp6Progress.currentDomain===d.name ? "#004488" : "#b8cce8",
                  color:      exp6Progress.currentDomain===d.name ? "#fff"    : "#002244",
                  borderRadius:2, fontSize:5, fontWeight:700}}>
                  {d.emoji}
                </div>
              ))}
            </div>
            {exp6Progress.phase === 'training' ? (
              <div>
                <div style={{marginBottom:2}}>
                  Train trial: <span style={{float:"right", fontWeight:700}}>{exp6Progress.trainTrial ?? '—'}/{exp6Progress.trainingTrials ?? 10}</span>
                </div>
                <div style={{marginBottom:2}}>
                  Patterns: <span style={{float:"right", fontWeight:700, color:"#004488"}}>{exp6Progress.trainedPatterns ?? 0}</span>
                </div>
              </div>
            ) : (
              <div>
                <div style={{marginBottom:2}}>
                  Domain: <span style={{float:"right", fontWeight:700, maxWidth:"55%", overflow:"hidden", textOverflow:"ellipsis", whiteSpace:"nowrap"}}>{exp6Progress.domainLabel ?? '—'}</span>
                </div>
                <div style={{marginBottom:2}}>
                  Condition: <span style={{float:"right", fontWeight:700}}>{exp6Progress.currentCondition ?? '—'}</span>
                </div>
                <div style={{marginBottom:2}}>
                  Trained patterns: <span style={{float:"right", fontWeight:700, color:"#004488"}}>{exp6Progress.totalTrainedPatterns ?? 0}</span>
                </div>
              </div>
            )}
            <div style={{marginTop:2, borderTop:"1px solid #6699cc", paddingTop:2}}>
              Complete: <span style={{float:"right", fontWeight:700}}>{exp6Progress.percentComplete ?? '0.0'}%</span>
            </div>
          </div>
        )}

        {/* ── Experiment 6 idle hint ── */}
        {experimentMode && selectedExp===6 && !exp6Progress && (
          <div style={{fontSize:6, padding:"8px", background:"#eef4ff", borderRadius:3, border:"1px solid #99bbdd", color:"#002244"}}>
            <strong>🏭 Warehouse</strong> — source control (same domain)<br/>
            <strong>🚀 Mars Gravity</strong> — 0.38g physics transfer<br/>
            <strong>📡 High Noise</strong> — 20% sensor degradation<br/>
            <strong>⚡ Speed</strong> — double-reward objective<br/>
            <strong>🛡️ Safety</strong> — wall-bounce penalty<br/>
            Conditions: A (no memory) vs frozen_D (no new consolidation).<br/>
            Transfer efficiency = target advantage / source advantage × 100%.<br/>
            Results auto-download as JSON.
          </div>
        )}

        {/* ── Experiment 8 controls ── */}
        {experimentMode && selectedExp===8 && (
          <div>
            <div style={{fontSize:6, color:"#1a2e00", marginBottom:4, lineHeight:1.6}}>
              <strong>Weight Optimization</strong><br/>
              10 configs × (10 train + 5 obj × 2 cond × 3 test) = <strong>400 trials</strong><br/>
              Beats +11.27%? T-test vs baseline_equal.
            </div>
            {exp8Status !== 'running' ? (
              <button onClick={handleRunExp8} style={{width:"100%", padding:"9px 4px", fontSize:8, letterSpacing:2, fontFamily:"inherit", background:"#224400", color:"#ffffff", border:"none", borderRadius:3, cursor:"pointer"}}>
                ▶ RUN EXP 8 (400 trials)
              </button>
            ) : (
              <button onClick={handleStopExp8} style={{width:"100%", padding:"9px 4px", fontSize:8, letterSpacing:2, fontFamily:"inherit", background:"#ff5533", color:"#ffffff", border:"none", borderRadius:3, cursor:"pointer"}}>
                ■ ABORT EXP 8
              </button>
            )}
          </div>
        )}

        {/* ── Experiment 8 progress panel ── */}
        {experimentMode && selectedExp===8 && exp8Progress && (
          <div style={{fontSize:6, padding:"8px", background:"#f0f8e8", borderRadius:3, border:"1px solid #88bb44"}}>
            <div style={{color:"#224400", fontWeight:700, marginBottom:4}}>
              {exp8Status==='complete' ? '✓ EXP 8 COMPLETE — JSON saved' : 'EXP 8 RUNNING…'}
            </div>
            <div style={{marginBottom:2}}>
              Trials: <span style={{float:"right", fontWeight:700}}>{exp8Progress.completedTrials ?? 0}/{exp8Progress.totalTrials ?? 400}</span>
            </div>
            <div style={{height:5, background:"#c8e8a0", borderRadius:2, marginBottom:3}}>
              <div style={{height:"100%", borderRadius:2, background:"#224400",
                width:`${exp8Progress.percentComplete ?? 0}%`, transition:"width 0.4s"}}/>
            </div>
            {/* Phase badges */}
            <div style={{display:"flex", gap:2, marginBottom:3}}>
              {['training','testing'].map(p => (
                <div key={p} style={{flex:1, textAlign:"center", padding:"2px 3px",
                  background: exp8Progress.phase===p ? "#224400" : "#c8e8a0",
                  color:      exp8Progress.phase===p ? "#fff"    : "#1a2e00",
                  borderRadius:2, fontSize:5, fontWeight:700}}>
                  {p==='training' ? '⚙️ TRAIN' : '🔬 TEST'}
                </div>
              ))}
            </div>
            {/* Hypothesis badges */}
            <div style={{display:"flex", gap:2, marginBottom:3, flexWrap:"wrap"}}>
              {['baseline','speed','efficiency','robustness','compound','food'].map(h => (
                <div key={h} style={{flex:"0 0 auto", padding:"2px 4px",
                  background: exp8Progress.hypothesis===h ? "#224400" : "#c8e8a0",
                  color:      exp8Progress.hypothesis===h ? "#fff"    : "#1a2e00",
                  borderRadius:2, fontSize:4.5, fontWeight:700}}>
                  {h}
                </div>
              ))}
            </div>
            {exp8Progress.phase === 'training' ? (
              <div>
                <div style={{marginBottom:2}}>
                  Config: <span style={{float:"right", fontWeight:700, maxWidth:"55%", overflow:"hidden", textOverflow:"ellipsis", whiteSpace:"nowrap"}}>{exp8Progress.configLabel ?? '—'}</span>
                </div>
                <div style={{marginBottom:2}}>
                  Train trial: <span style={{float:"right", fontWeight:700}}>{exp8Progress.trainTrial ?? '—'}/{exp8Progress.trainingTrials ?? 10}</span>
                </div>
                <div style={{marginBottom:2}}>
                  Patterns: <span style={{float:"right", fontWeight:700, color:"#224400"}}>{exp8Progress.trainedPatterns ?? 0}</span>
                </div>
              </div>
            ) : (
              <div>
                <div style={{marginBottom:2}}>
                  Config: <span style={{float:"right", fontWeight:700, maxWidth:"50%", overflow:"hidden", textOverflow:"ellipsis", whiteSpace:"nowrap"}}>{exp8Progress.configLabel ?? '—'}</span>
                </div>
                <div style={{marginBottom:2}}>
                  Objective: <span style={{float:"right", fontWeight:700}}>{exp8Progress.currentObjective ?? '—'}</span>
                </div>
                <div style={{marginBottom:2}}>
                  Condition: <span style={{float:"right", fontWeight:700}}>{exp8Progress.currentCondition ?? '—'}</span>
                </div>
              </div>
            )}
            <div style={{marginTop:2, borderTop:"1px solid #88bb44", paddingTop:2}}>
              Complete: <span style={{float:"right", fontWeight:700}}>{exp8Progress.percentComplete ?? '0.0'}%</span>
            </div>
          </div>
        )}

        {/* ── Experiment 8 idle hint ── */}
        {experimentMode && selectedExp===8 && !exp8Progress && (
          <div style={{fontSize:6, padding:"8px", background:"#f4fce8", borderRadius:3, border:"1px solid #aabb77", color:"#1a2e00"}}>
            <strong>H1 Speed</strong> — food:30/40, reduce efficiency<br/>
            <strong>H2 Efficiency</strong> — balance:30/40, reduce eff<br/>
            <strong>H3 Robustness</strong> — accuracy+balance up, speed down<br/>
            <strong>H4 Compound</strong> — food+balance up, mixed reductions<br/>
            <strong>H5 Baseline</strong> — equal 20/20/20/20/20 (control)<br/>
            T-test each config vs baseline_equal.<br/>
            Winner if improvement &gt; 0.5% AND p &lt; 0.05.<br/>
            JSON includes _ranking, _tTests, _conclusion.
          </div>
        )}

        {/* ── Experiment 9 controls ── */}
        {experimentMode && selectedExp===9 && (
          <div>
            <div style={{fontSize:6, color:"#002233", marginBottom:4, lineHeight:1.6}}>
              <strong>Learning Dynamics</strong><br/>
              6 checkpoints × 2 reps × 20 tests = <strong>394 trials</strong><br/>
              Curve shape, convergence, overfitting check.
            </div>
            {exp9Status !== 'running' ? (
              <button onClick={handleRunExp9} style={{width:"100%", padding:"9px 4px", fontSize:8, letterSpacing:2, fontFamily:"inherit", background:"#005566", color:"#ffffff", border:"none", borderRadius:3, cursor:"pointer"}}>
                ▶ RUN EXP 9 (394 trials)
              </button>
            ) : (
              <button onClick={handleStopExp9} style={{width:"100%", padding:"9px 4px", fontSize:8, letterSpacing:2, fontFamily:"inherit", background:"#ff5533", color:"#ffffff", border:"none", borderRadius:3, cursor:"pointer"}}>
                ■ ABORT EXP 9
              </button>
            )}
          </div>
        )}

        {/* ── Experiment 9 progress panel ── */}
        {experimentMode && selectedExp===9 && exp9Progress && (
          <div style={{fontSize:6, padding:"8px", background:"#e8f8ff", borderRadius:3, border:"1px solid #55aacc"}}>
            <div style={{color:"#005566", fontWeight:700, marginBottom:4}}>
              {exp9Status==='complete' ? '✓ EXP 9 COMPLETE — JSON saved' : 'EXP 9 RUNNING…'}
            </div>
            <div style={{marginBottom:2}}>
              Trials: <span style={{float:"right", fontWeight:700}}>{exp9Progress.completedTrials ?? 0}/{exp9Progress.totalTrials ?? 394}</span>
            </div>
            <div style={{height:5, background:"#aaddef", borderRadius:2, marginBottom:3}}>
              <div style={{height:"100%", borderRadius:2, background:"#005566",
                width:`${exp9Progress.percentComplete ?? 0}%`, transition:"width 0.4s"}}/>
            </div>
            {/* Phase badges */}
            <div style={{display:"flex", gap:2, marginBottom:3}}>
              {['training','testing'].map(p => (
                <div key={p} style={{flex:1, textAlign:"center", padding:"2px 3px",
                  background: exp9Progress.phase===p ? "#005566" : "#aaddef",
                  color:      exp9Progress.phase===p ? "#fff"    : "#002233",
                  borderRadius:2, fontSize:5, fontWeight:700}}>
                  {p==='training' ? '⚙️ TRAIN' : '🔬 TEST'}
                </div>
              ))}
            </div>
            {/* Checkpoint badges */}
            <div style={{display:"flex", gap:2, marginBottom:3}}>
              {[0,2,5,10,20,40].map(cp => (
                <div key={cp} style={{flex:1, textAlign:"center", padding:"2px 2px",
                  background: exp9Progress.checkpoint===cp ? "#005566" : "#aaddef",
                  color:      exp9Progress.checkpoint===cp ? "#fff"    : "#002233",
                  borderRadius:2, fontSize:4.5, fontWeight:700}}>
                  {cp===0 ? '∅' : cp}
                </div>
              ))}
            </div>
            {exp9Progress.phase === 'training' ? (
              <div>
                <div style={{marginBottom:2}}>
                  Checkpoint: <span style={{float:"right", fontWeight:700}}>{exp9Progress.checkpoint} trials</span>
                </div>
                <div style={{marginBottom:2}}>
                  Rep: <span style={{float:"right", fontWeight:700}}>{(exp9Progress.rep ?? 0) + 1} / 2</span>
                </div>
                <div style={{marginBottom:2}}>
                  Train trial: <span style={{float:"right", fontWeight:700}}>{exp9Progress.trainTrial ?? '—'}/{exp9Progress.totalTrainTrials ?? '—'}</span>
                </div>
                <div style={{marginBottom:2}}>
                  Patterns: <span style={{float:"right", fontWeight:700, color:"#005566"}}>{exp9Progress.trainedPatterns ?? 0}</span>
                </div>
              </div>
            ) : (
              <div>
                <div style={{marginBottom:2}}>
                  Checkpoint: <span style={{float:"right", fontWeight:700}}>{exp9Progress.checkpoint} trials · rep {(exp9Progress.rep ?? 0) + 1}/2</span>
                </div>
                <div style={{marginBottom:2}}>
                  Objective: <span style={{float:"right", fontWeight:700}}>{exp9Progress.currentObjective ?? '—'}</span>
                </div>
                <div style={{marginBottom:2}}>
                  Condition: <span style={{float:"right", fontWeight:700}}>{exp9Progress.currentCondition ?? '—'}</span>
                </div>
                <div style={{marginBottom:2}}>
                  Patterns avail.: <span style={{float:"right", fontWeight:700, color:"#005566"}}>{exp9Progress.totalTrainedPatterns ?? 0}</span>
                </div>
              </div>
            )}
            <div style={{marginTop:2, borderTop:"1px solid #55aacc", paddingTop:2}}>
              Complete: <span style={{float:"right", fontWeight:700}}>{exp9Progress.percentComplete ?? '0.0'}%</span>
            </div>
          </div>
        )}

        {/* ── Experiment 9 idle hint ── */}
        {experimentMode && selectedExp===9 && !exp9Progress && (
          <div style={{fontSize:6, padding:"8px", background:"#eef8ff", borderRadius:3, border:"1px solid #88ccdd", color:"#002233"}}>
            <strong>∅ 0 trials</strong> — pure reactive baseline<br/>
            <strong>2 trials</strong> — very early learning<br/>
            <strong>5 trials</strong> — quarter-way to plateau<br/>
            <strong>10 trials</strong> — half-way learning<br/>
            <strong>20 trials</strong> — near convergence<br/>
            <strong>40 trials</strong> — overfitting check<br/>
            Curve classified: exponential / sigmoid / linear.<br/>
            JSON: _advantageCurve, _curveType, _convergencePoint,<br/>
            _deploymentGuidance, _overfitting.
          </div>
        )}

        {/* ── Experiment 10 controls ── */}
        {experimentMode && selectedExp===10 && (
          <div>
            <div style={{fontSize:6, color:"#660088", marginBottom:4, lineHeight:1.6}}>
              <strong>Exp 10a</strong> — Reward Signal Coherence<br/>
              Collision penalty: −0.5 per bounce (accuracy variant)<br/>
              Control: EXP5 training + EXP5.5 terminal scoring<br/>
              Coherent: EXP10 training + EXP10 terminal scoring
            </div>
            {exp10Status !== 'running' ? (
              <button onClick={handleRunExp10} style={{width:"100%", padding:"9px 4px", fontSize:8, letterSpacing:2, fontFamily:"inherit", background:"#330044", color:"#ffffff", border:"none", borderRadius:3, cursor:"pointer"}}>
                ▶ RUN EXP 10a (160 trials)
              </button>
            ) : (
              <button onClick={handleStopExp10} style={{width:"100%", padding:"9px 4px", fontSize:8, letterSpacing:2, fontFamily:"inherit", background:"#660066", color:"#ffffff", border:"none", borderRadius:3, cursor:"pointer"}}>
                ■ ABORT EXP 10a
              </button>
            )}
          </div>
        )}

        {/* ── Experiment 10 progress panel ── */}
        {experimentMode && selectedExp===10 && exp10Progress && (
          <div style={{fontSize:6, padding:"8px", background:"#f9eeff", borderRadius:3, border:"1px solid #cc88ff"}}>
            <div style={{color:"#330044", fontWeight:700, marginBottom:4}}>
              {exp10Status==='complete' ? '✓ EXP 10a COMPLETE — EXP10A_RESULTS.json saved' : 'EXP 10a RUNNING…'}
            </div>
            <div style={{marginBottom:2}}>
              Trials: <span style={{float:"right", fontWeight:700}}>{exp10Progress.completedTrials ?? 0}/{exp10Progress.totalTrials ?? 160}</span>
            </div>
            <div style={{height:4, background:"#ddd", borderRadius:2, marginBottom:4}}>
              <div style={{height:"100%", borderRadius:2, background:"#330044",
                width:`${exp10Progress.percentComplete ?? 0}%`, transition:"width 0.4s"}}/>
            </div>
            <div style={{display:"flex", gap:2, marginBottom:4}}>
              {['training','testing'].map(p => (
                <div key={p} style={{flex:1, textAlign:"center", padding:"2px 1px", borderRadius:2, fontSize:5,
                  background: exp10Progress.phase===p ? "#330044" : "#ddbbee",
                  color:      exp10Progress.phase===p ? "#fff"    : "#220033"}}>
                  {p}
                </div>
              ))}
            </div>
            <div style={{display:"flex", gap:2, marginBottom:3}}>
              {['control','coherent'].map(g => (
                <div key={g} style={{flex:1, textAlign:"center", padding:"2px 1px", borderRadius:2, fontSize:5,
                  background: exp10Progress.group===g ? "#550066" : "#ddbbee",
                  color:      exp10Progress.group===g ? "#fff"    : "#220033"}}>
                  {g}
                </div>
              ))}
            </div>
            {exp10Progress.phase === 'training' ? (
              <div>
                <div style={{marginBottom:1}}>Variant: <span style={{float:"right", fontWeight:700}}>{exp10Progress.variant ?? '—'}</span></div>
                <div style={{marginBottom:1}}>Train trial: <span style={{float:"right", fontWeight:700}}>{exp10Progress.trainingTrial ?? '—'}/{exp10Progress.totalTrainingTrials ?? '—'}</span></div>
                <div style={{marginBottom:1}}>Patterns: <span style={{float:"right", fontWeight:700, color:"#330044"}}>{exp10Progress.ltmPatterns ?? 0}</span></div>
              </div>
            ) : (
              <div>
                <div style={{marginBottom:1}}>Variant: <span style={{float:"right", fontWeight:700}}>{exp10Progress.variant ?? '—'}</span></div>
                <div style={{marginBottom:1}}>Condition: <span style={{float:"right", fontWeight:700}}>{exp10Progress.currentCondition ?? '—'}</span></div>
                <div style={{marginBottom:1}}>Patterns avail.: <span style={{float:"right", fontWeight:700, color:"#330044"}}>{exp10Progress.totalTrainedPatterns ?? 0}</span></div>
              </div>
            )}
            <div style={{marginTop:3, borderTop:"1px solid #cc88ff", paddingTop:3}}>
              Complete: <span style={{float:"right", fontWeight:700}}>{exp10Progress.percentComplete ?? '0.0'}%</span>
            </div>
          </div>
        )}

        {/* ── Experiment 10 idle hint ── */}
        {experimentMode && selectedExp===10 && !exp10Progress && (
          <div style={{fontSize:6, padding:"8px", background:"#f5eeff", borderRadius:3, border:"1px solid #cc88ff", color:"#220033"}}>
            <strong>Coherence hypothesis:</strong><br/>
            Training + test reward signals aligned → higher D-vs-A advantage.<br/>
            <strong>5 variants</strong>: baseline, efficiency, accuracy, speed, balance<br/>
            <strong>Accuracy (Exp 10a)</strong>: −0.5 per wall bounce in training.<br/>
            Control trains without collision penalty but scores with it.<br/>
            JSON: _coherenceDelta, _avgGeneralizationIndex per group,<br/>
            _conclusion, _collisionPenalty, _experimentVersion.
          </div>
        )}

        {/* ── Experiment 10b controls ── */}
        {experimentMode && selectedExp==='10b' && (
          <div>
            <div style={{fontSize:6, color:"#880033", marginBottom:4, lineHeight:1.6}}>
              <strong>Exp 10b</strong> — Collision Penalty Calibration<br/>
              Sweeps −1.0 / −2.0 / −5.0 vs reactive baseline (condition A)<br/>
              Key metric: safety_score = accuracy_adv × 0.4 + bounce_reduction × 0.6<br/>
              ISO/TS 15066: bounces ≈ hardware force/speed-limiting triggers
            </div>
            {exp10bStatus !== 'running' ? (
              <button onClick={handleRunExp10b} style={{width:"100%", padding:"9px 4px", fontSize:8, letterSpacing:2, fontFamily:"inherit", background:"#550022", color:"#ffffff", border:"none", borderRadius:3, cursor:"pointer"}}>
                ▶ RUN EXP 10b (120 trials)
              </button>
            ) : (
              <button onClick={handleStopExp10b} style={{width:"100%", padding:"9px 4px", fontSize:8, letterSpacing:2, fontFamily:"inherit", background:"#990033", color:"#ffffff", border:"none", borderRadius:3, cursor:"pointer"}}>
                ■ ABORT EXP 10b
              </button>
            )}
          </div>
        )}

        {/* ── Experiment 10b progress panel ── */}
        {experimentMode && selectedExp==='10b' && exp10bProgress && (
          <div style={{fontSize:6, padding:"8px", background:"#fff0f3", borderRadius:3, border:"1px solid #ffaaaa"}}>
            <div style={{color:"#550022", fontWeight:700, marginBottom:4}}>
              {exp10bStatus==='complete' ? '✓ EXP 10b COMPLETE — EXP10B_RESULTS.json saved' : 'EXP 10b RUNNING…'}
            </div>
            <div style={{marginBottom:2}}>
              Trials: <span style={{float:"right", fontWeight:700}}>{exp10bProgress.completedTrials ?? 0}/{exp10bProgress.totalTrials ?? 120}</span>
            </div>
            <div style={{height:4, background:"#ddd", borderRadius:2, marginBottom:4}}>
              <div style={{height:"100%", borderRadius:2, background:"#550022",
                width:`${exp10bProgress.percentComplete ?? 0}%`, transition:"width 0.4s"}}/>
            </div>
            <div style={{display:"flex", gap:2, marginBottom:3}}>
              {['training','testing'].map(p => (
                <div key={p} style={{flex:1, textAlign:"center", padding:"2px 1px", borderRadius:2, fontSize:5,
                  background: exp10bProgress.phase===p ? "#550022" : "#ffcccc",
                  color:      exp10bProgress.phase===p ? "#fff"    : "#440011"}}>
                  {p}
                </div>
              ))}
            </div>
            <div style={{marginBottom:3}}>
              Penalty: <span style={{float:"right", fontWeight:700, color:"#880033"}}>{exp10bProgress.penaltyLabel ?? '—'} ({exp10bProgress.penalty ?? '—'})</span>
            </div>
            {exp10bProgress.phase === 'training' ? (
              <div>
                <div style={{marginBottom:1}}>Train trial: <span style={{float:"right", fontWeight:700}}>{exp10bProgress.trainingTrial ?? '—'}/{exp10bProgress.totalTrainingTrials ?? '—'}</span></div>
                <div style={{marginBottom:1}}>Patterns: <span style={{float:"right", fontWeight:700, color:"#550022"}}>{exp10bProgress.ltmPatterns ?? 0}</span></div>
              </div>
            ) : (
              <div>
                <div style={{marginBottom:1}}>Objective: <span style={{float:"right", fontWeight:700}}>{exp10bProgress.currentObjective ?? '—'}</span></div>
                <div style={{marginBottom:1}}>Condition: <span style={{float:"right", fontWeight:700}}>{exp10bProgress.currentCondition ?? '—'}</span></div>
                <div style={{marginBottom:1}}>Patterns: <span style={{float:"right", fontWeight:700, color:"#550022"}}>{exp10bProgress.totalTrainedPatterns ?? 0}</span></div>
              </div>
            )}
            <div style={{marginTop:3, borderTop:"1px solid #ffaaaa", paddingTop:3}}>
              Complete: <span style={{float:"right", fontWeight:700}}>{exp10bProgress.percentComplete ?? '0.0'}%</span>
            </div>
          </div>
        )}

        {/* ── Experiment 10b idle hint ── */}
        {experimentMode && selectedExp==='10b' && !exp10bProgress && (
          <div style={{fontSize:6, padding:"8px", background:"#fff8f9", borderRadius:3, border:"1px solid #ffbbcc", color:"#440011"}}>
            <strong>Safety/performance tradeoff sweep</strong><br/>
            −1.0: light deterrent (double Exp 10a)<br/>
            −2.0: moderate deterrent (strong but preserving)<br/>
            −5.0: near-absolute deterrent (upper bound)<br/>
            <strong>safety_score</strong> = accuracy_adv × 0.4 + bounce_reduction × 0.6<br/>
            Bounce reduction → fewer ISO/TS 15066 HW triggers.<br/>
            JSON: _penaltyComparison, _optimalPenalty, _safetyScore,<br/>
            _bounceReductionPct per objective, _isoNote.
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
          {experimentMode && selectedExp==='10b'
            ? <><span style={{color:"#550022"}}>▶</span> 120 trials, 3 penalty levels<br/><span style={{color:"#550022"}}>↓</span> Saves EXP10B_RESULTS.json<br/><span style={{color:"#550022"}}>✓</span> ~6–10 min runtime</>
            : experimentMode && selectedExp===10
            ? <><span style={{color:"#330044"}}>▶</span> 160 trials, 2 groups × 5 variants<br/><span style={{color:"#330044"}}>↓</span> Saves EXP10A_RESULTS.json<br/><span style={{color:"#330044"}}>✓</span> ~8–12 min runtime</>
            : experimentMode && selectedExp===9
            ? <><span style={{color:"#005566"}}>▶</span> 394 trials, 6 checkpoints<br/><span style={{color:"#005566"}}>↓</span> Auto-downloads JSON<br/><span style={{color:"#005566"}}>✓</span> ~30–35 min runtime</>
            : experimentMode && selectedExp===8
            ? <><span style={{color:"#224400"}}>▶</span> 400 trials, 10 weight configs<br/><span style={{color:"#224400"}}>↓</span> Auto-downloads JSON<br/><span style={{color:"#224400"}}>✓</span> ~35–45 min runtime</>
            : experimentMode && selectedExp===6
            ? <><span style={{color:"#004488"}}>▶</span> 60 trials, 5 transfer domains<br/><span style={{color:"#004488"}}>↓</span> Auto-downloads JSON<br/><span style={{color:"#004488"}}>✓</span> ~25–35 min runtime</>
            : experimentMode && selectedExp==='5.5.5'
            ? <><span style={{color:"#770077"}}>▶</span> 300 trials, 5 weight combos<br/><span style={{color:"#770077"}}>↓</span> Auto-downloads JSON<br/><span style={{color:"#770077"}}>✓</span> ~25–35 min runtime</>
            : experimentMode && selectedExp===5.5
            ? <><span style={{color:"#5500aa"}}>▶</span> 120 trials, 2 training variants<br/><span style={{color:"#5500aa"}}>↓</span> Auto-downloads JSON<br/><span style={{color:"#5500aa"}}>✓</span> ~15–20 min runtime</>
            : experimentMode && selectedExp===5
            ? <><span style={{color:"#884400"}}>▶</span> 50 runs, 5 reward variants<br/><span style={{color:"#884400"}}>↓</span> Auto-downloads JSON<br/><span style={{color:"#884400"}}>✓</span> ~5–8 min runtime</>
            : experimentMode && selectedExp===4.5
            ? <><span style={{color:"#007755"}}>▶</span> 160 trials, 2 variants<br/><span style={{color:"#007755"}}>↓</span> Auto-downloads JSON<br/><span style={{color:"#007755"}}>✓</span> ~10–15 min runtime</>
            : experimentMode && selectedExp===4
            ? <><span style={{color:"#006644"}}>▶</span> 220 trials, 2 phases<br/><span style={{color:"#006644"}}>↓</span> Auto-downloads JSON<br/><span style={{color:"#006644"}}>✓</span> ~10–15 min runtime</>
            : experimentMode && selectedExp===3
            ? <><span style={{color:"#cc6600"}}>▶</span> 900 runs, 3 stressor profiles<br/><span style={{color:"#cc6600"}}>↓</span> Auto-downloads JSON<br/><span style={{color:"#cc6600"}}>✓</span> ~20–25 min runtime</>
            : experimentMode && selectedExp===2
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