/**
 * Utility functions for processing and transforming quantum pattern data for visualization
 */

/**
 * Process quantum states for visualization
 * @param {Array} patterns - Array of quantum patterns
 * @returns {Array} Processed quantum states with visualization-ready components
 */
export const processQuantumStates = (patterns) => {
  return patterns.map(pattern => ({
    ...pattern,
    stateComponents: Array.from(
      { length: pattern.quantum_state?.length || 0 },
      (_, i) => ({
        index: i,
        amplitude: pattern.quantum_state?.[i] || 0,
        phase: Math.atan2(
          pattern.quantum_state?.[i]?.imag || 0,
          pattern.quantum_state?.[i]?.real || 0
        )
      })
    )
  }));
};

/**
 * Format correlation data for heatmap visualization
 * @param {Array} correlationMatrix - 2D array of correlation values
 * @returns {Array} Formatted data for heatmap visualization
 */
export const formatCorrelationData = (correlationMatrix) => {
  return correlationMatrix.map((row, i) => 
    row.map((value, j) => ({
      x: i,
      y: j,
      value: value,
      label: `(${i},${j}): ${value.toFixed(3)}`
    }))
  ).flat();
};

/**
 * Process temporal data for time series visualization
 * @param {Object} data - Raw temporal data
 * @param {Array} patterns - Array of quantum patterns
 * @returns {Array} Processed data for time series visualization
 */
export const processTemporalData = (data, patterns) => {
  const timePoints = Object.keys(data.temporal || {}).sort();
  return timePoints.map(time => {
    const point = {
      time: parseFloat(time),
      ...data.numerical[time]
    };

    // Add pattern strengths at this time point
    patterns.forEach(pattern => {
      if (pattern.temporal_span) {
        const [start, end] = pattern.temporal_span;
        if (point.time >= start && point.time <= end) {
          point[`${pattern.modality}_strength`] = pattern.strength;
        }
      }
    });

    return point;
  });
};

/**
 * Calculate quantum state statistics
 * @param {Array} quantumStates - Array of quantum states
 * @returns {Object} Statistical metrics for quantum states
 */
export const calculateQuantumStateStats = (quantumStates) => {
  const stats = quantumStates.map(state => {
    const amplitudes = state.stateComponents.map(c => c.amplitude);
    return {
      modality: state.modality,
      mean: amplitudes.reduce((a, b) => a + b, 0) / amplitudes.length,
      max: Math.max(...amplitudes),
      min: Math.min(...amplitudes),
      variance: amplitudes.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / amplitudes.length
    };
  });

  return stats;
};

/**
 * Generate color scale for quantum states
 * @param {number} numStates - Number of quantum states
 * @returns {Array} Array of color values
 */
export const generateQuantumStateColors = (numStates) => {
  return Array.from({ length: numStates }, (_, i) => ({
    color: `hsl(${i * 360 / numStates}, 70%, 50%)`,
    index: i
  }));
};

/**
 * Format pattern details for display
 * @param {Object} pattern - Quantum pattern object
 * @returns {Object} Formatted pattern details
 */
export const formatPatternDetails = (pattern) => {
  return {
    basic: {
      modality: pattern.modality,
      type: pattern.pattern_type,
      strength: pattern.strength.toFixed(3)
    },
    quantum: Object.entries(pattern.quantum_metrics || {}).reduce(
      (acc, [key, value]) => ({
        ...acc,
        [key]: typeof value === 'number' ? value.toFixed(3) : value
      }),
      {}
    ),
    temporal: pattern.temporal_span ? {
      start: pattern.temporal_span[0],
      end: pattern.temporal_span[1],
      duration: pattern.temporal_span[1] - pattern.temporal_span[0]
    } : null,
    correlations: Object.entries(pattern.correlations || {}).reduce(
      (acc, [id, value]) => ({
        ...acc,
        [id]: typeof value === 'number' ? value.toFixed(3) : value
      }),
      {}
    )
  };
};

/**
 * Calculate visualization dimensions based on container size
 * @param {number} containerWidth - Width of container
 * @param {number} containerHeight - Height of container
 * @returns {Object} Calculated dimensions for various visualization components
 */
export const calculateVisualizationDimensions = (containerWidth, containerHeight) => {
  return {
    chart: {
      width: Math.max(containerWidth * 0.9, 300),
      height: Math.max(containerHeight * 0.6, 200)
    },
    heatmap: {
      width: Math.max(containerWidth * 0.45, 250),
      height: Math.max(containerWidth * 0.45, 250)
    },
    details: {
      width: Math.max(containerWidth * 0.9, 300),
      height: Math.max(containerHeight * 0.3, 150)
    }
  };
};

/**
 * Generate axis configurations for different visualization types
 * @param {string} visualizationType - Type of visualization
 * @returns {Object} Axis configuration object
 */
export const getAxisConfig = (visualizationType) => {
  const configs = {
    quantum_states: {
      xAxis: {
        label: 'State Component',
        type: 'number',
        domain: [0, 'auto']
      },
      yAxis: {
        label: 'Amplitude',
        type: 'number',
        domain: [-1, 1]
      }
    },
    temporal: {
      xAxis: {
        label: 'Time',
        type: 'number',
        domain: ['auto', 'auto']
      },
      yAxis: {
        label: 'Value',
        type: 'number',
        domain: ['auto', 'auto']
      }
    },
    correlation: {
      xAxis: {
        label: 'Pattern Index',
        type: 'number',
        domain: [0, 'auto']
      },
      yAxis: {
        label: 'Pattern Index',
        type: 'number',
        domain: [0, 'auto']
      }
    }
  };

  return configs[visualizationType] || configs.quantum_states;
}; 