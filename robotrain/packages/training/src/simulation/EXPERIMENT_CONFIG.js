/**
 * EXPERIMENT_CONFIG.js — re-export from the canonical location.
 *
 * This shim lets files in src/components/ and src/ use a short import path:
 *
 *   // From src/components/ExperimentRunner_v2.js:
 *   import { SIMULATION_CONFIG, ... } from './EXPERIMENT_CONFIG.js';
 *
 *   // From src/App.jsx:
 *   import { SIMULATION_CONFIG, ... } from './components/EXPERIMENT_CONFIG.js';
 *
 * All values live in src/components/memory/EXPERIMENT_CONFIG.js — edit there.
 */
export * from './memory/EXPERIMENT_CONFIG.js';
