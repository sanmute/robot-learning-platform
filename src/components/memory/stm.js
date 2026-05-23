/**
 * STM.js - Short-Term Memory Buffer for Embodied Agents
 * 
 * Implements a circular buffer that captures recent agent experience:
 * sensory states, neural activations, actions, rewards, and attention weights.
 * 
 * Used in v0.10+ for:
 * - Real-time feedback to STM
 * - Sequence extraction for consolidation
 * - Decay-weighted access to recent history
 * 
 * Author: Santeri
 * Version: 1.0
 * Date: May 2026
 */

/**
 * STMFrame - Single frame of agent experience
 * 
 * Represents a snapshot of the agent at one point in time.
 * Includes sensory input, internal state, action taken, and resulting reward.
 */
class STMFrame {
  constructor(timestamp, sensoryState, neuralState, action, reward, attentionWeights) {
    // Time identifier (frame number or millisecond timestamp)
    this.timestamp = timestamp;

    // [10] binary vector from 5 sensors (each: obstacle + food)
    // sensoryState[0-4] = obstacle sensors (5 rays)
    // sensoryState[5-9] = food sensors (5 rays)
    this.sensoryState = sensoryState;

    // [25] neural activations from Hopfield network
    // Represents internal state at this moment
    this.neuralState = neuralState;

    // Action taken: {left: boolean, forward: boolean, right: boolean}
    this.action = action;

    // Scalar reward received after action
    this.reward = reward;

    // [M] attention weights from Hopfield recall
    // Shows which stored patterns were active
    this.attentionWeights = attentionWeights;

    // For future surprise calculation
    this.predictionTarget = null;
  }

  /**
   * Get age of this frame in frames (at 60fps)
   * @param {number} currentTimestamp - Current frame timestamp
   * @returns {number} Age in frames
   */
  getAge(currentTimestamp) {
    return currentTimestamp - this.timestamp;
  }

  /**
   * Simple string representation for debugging
   * @returns {string}
   */
  toString() {
    return `STMFrame(t=${this.timestamp}, reward=${this.reward.toFixed(1)}, action=${Object.keys(this.action).filter(k => this.action[k]).join('|')})`;
  }
}

/**
 * ShortTermMemory - Circular buffer of recent experience
 * 
 * Maintains a fixed-size window of recent STM frames.
 * Older frames are automatically overwritten.
 * Provides time-decay weighting for accessing older frames.
 * 
 * Parameters:
 * - capacity: Number of frames to store (default 60 ≈ 1 second at 60fps)
 * - tauDecay: Time constant in seconds (default 0.5s)
 */
class ShortTermMemory {
  constructor(capacity = 60, tauDecay = 0.5) {
    // Maximum frames to store
    this.capacity = capacity;

    // Array holding STMFrame objects (dynamic size up to capacity)
    this.buffer = [];

    // Index for circular buffer (next position to write)
    this.currentIndex = 0;

    // Flag: is buffer at full capacity?
    this.isFull = false;

    // Time constant for exponential decay (in seconds)
    // weight(t-k) = exp(-k / tau_decay)
    this.tauDecay = tauDecay;

    // Metadata
    this.createdAt = Date.now();
    this.framesAdded = 0;
  }

  /**
   * Add a frame to STM buffer
   * 
   * If buffer not full: append to array
   * If buffer full: overwrite oldest entry (at currentIndex)
   * 
   * @param {STMFrame} frame - Frame to add
   */
  add(frame) {
    if (this.buffer.length < this.capacity) {
      // Buffer not yet full, just append
      this.buffer.push(frame);
    } else {
      // Buffer full, overwrite oldest (circular)
      this.isFull = true;
      this.buffer[this.currentIndex] = frame;
      this.currentIndex = (this.currentIndex + 1) % this.capacity;
    }
    
    this.framesAdded++;
  }

  /**
   * Retrieve frame at specific offset
   * 
   * offset = 0: current frame (most recent)
   * offset = -1: one frame ago
   * offset = -30: 30 frames ago (oldest in 60-frame window)
   * 
   * @param {number} offset - Negative offset from current
   * @returns {STMFrame|null} Frame at offset, or null if invalid
   */
  getFrame(offset) {
    // Validate offset
    if (offset > 0 || offset < -this.capacity) {
      return null;
    }

    if (this.buffer.length === 0) {
      return null;
    }

    // Calculate the index of the NEWEST frame:
    // - Not full: last pushed element sits at buffer.length - 1
    // - Full: currentIndex points to the NEXT slot to overwrite (oldest),
    //         so newest is one step behind it in the circular ring
    const actualCurrentIndex = this.isFull
      ? (this.currentIndex - 1 + this.capacity) % this.capacity
      : this.buffer.length - 1;
    
    // offset is negative, so we add it
    const idx = (actualCurrentIndex + offset + this.capacity) % this.capacity;
    
    return this.buffer[idx] || null;
  }

  /**
   * Get last N frames in chronological order
   * 
   * Most useful for sequence extraction (STM[t-29] to STM[t])
   * 
   * @param {number} lookback - Number of frames to retrieve
   * @returns {Array<STMFrame>} Array of frames, oldest first
   */
  getWindow(lookback) {
    const frames = [];
    
    // Start from oldest in window, go to most recent
    for (let i = lookback - 1; i >= 0; i--) {
      const frame = this.getFrame(-i);
      if (frame) {
        frames.push(frame);
      }
    }
    
    return frames;
  }

  /**
   * Get last N frames as object with descriptive keys
   * Useful for analysis and debugging
   * 
   * @param {number} lookback - Number of frames
   * @returns {Object} Object with frame metadata
   */
  getWindowAnalysis(lookback) {
    const frames = this.getWindow(lookback);
    
    return {
      count: frames.length,
      timespan: frames.length > 0 ? frames[frames.length - 1].timestamp - frames[0].timestamp : 0,
      totalReward: frames.reduce((sum, f) => sum + f.reward, 0),
      avgReward: frames.length > 0 ? frames.reduce((sum, f) => sum + f.reward, 0) / frames.length : 0,
      frames: frames
    };
  }

  /**
   * Calculate exponential decay weight for a frame based on age
   * 
   * weight(age) = exp(-age / tau_decay)
   * 
   * At age = tau_decay: weight ≈ 0.368 (1/e)
   * At age = 2*tau_decay: weight ≈ 0.135 (1/e²)
   * 
   * @param {number} frameIndex - Negative index (-0, -1, -30, etc.)
   * @returns {number} Decay weight [0, 1]
   */
  applyDecay(frameIndex) {
    const k = Math.abs(frameIndex);
    const tau = this.tauDecay * 60; // Convert seconds to frames (at 60fps)
    
    return Math.exp(-k / tau);
  }

  /**
   * Get weighted average of a property across recent frames
   * 
   * Useful for smoothing noisy signals
   * Example: weightedAverageReward = getWeightedAverage('reward', 30)
   * 
   * @param {string} property - Property name ('reward', etc.)
   * @param {number} lookback - How many frames to include
   * @returns {number} Weighted average
   */
  getWeightedAverage(property, lookback) {
    const frames = this.getWindow(lookback);
    
    if (frames.length === 0) return 0;
    
    let weightedSum = 0;
    let totalWeight = 0;
    
    for (let i = 0; i < frames.length; i++) {
      const frame = frames[i];
      const offset = -i; // offset from current
      const weight = this.applyDecay(offset);
      
      if (frame[property] !== undefined) {
        weightedSum += frame[property] * weight;
        totalWeight += weight;
      }
    }
    
    return totalWeight > 0 ? weightedSum / totalWeight : 0;
  }

  /**
   * Clear buffer completely
   * Useful for resetting between trials
   */
  clear() {
    this.buffer = [];
    this.currentIndex = 0;
    this.isFull = false;
    this.framesAdded = 0;
  }

  /**
   * Get statistics about current STM state
   * Useful for debugging and analysis
   * 
   * @returns {Object} Statistics object
   */
  getStats() {
    return {
      size: this.buffer.length,
      capacity: this.capacity,
      isFull: this.isFull,
      framesAdded: this.framesAdded,
      oldestTimestamp: this.buffer.length > 0 ? this.getFrame(-Math.min(this.capacity - 1, this.buffer.length - 1))?.timestamp : null,
      newestTimestamp: this.buffer.length > 0 ? this.getFrame(0)?.timestamp : null,
      avgRewardRecent: this.getWeightedAverage('reward', Math.min(10, this.buffer.length))
    };
  }

  /**
   * Get a summary of recent experience for logging
   * 
   * @returns {string} Human-readable summary
   */
  getSummary() {
    const stats = this.getStats();
    return `STM(${stats.size}/${stats.capacity}, avg_reward=${stats.avgRewardRecent.toFixed(1)})`;
  }

  /**
   * Export entire buffer as JSON
   * Useful for analysis and visualization
   * 
   * @returns {Object} JSON-serializable object
   */
  toJSON() {
    return {
      capacity: this.capacity,
      size: this.buffer.length,
      isFull: this.isFull,
      tauDecay: this.tauDecay,
      framesAdded: this.framesAdded,
      createdAt: this.createdAt,
      frames: this.buffer.map((frame, idx) => ({
        index: idx,
        timestamp: frame.timestamp,
        reward: frame.reward,
        sensoryState: frame.sensoryState,
        neuralState: frame.neuralState,
        action: frame.action,
        attentionWeights: frame.attentionWeights
      }))
    };
  }

  /**
   * Import STM from JSON
   * Useful for loading saved memories
   * 
   * @param {Object} data - JSON data from toJSON()
   * @static
   */
  static fromJSON(data) {
    const stm = new ShortTermMemory(data.capacity, data.tauDecay);
    
    data.frames.forEach(frameData => {
      const frame = new STMFrame(
        frameData.timestamp,
        frameData.sensoryState,
        frameData.neuralState,
        frameData.action,
        frameData.reward,
        frameData.attentionWeights
      );
      stm.add(frame);
    });
    
    return stm;
  }
}

// Export for use in other modules
export { STMFrame, ShortTermMemory };