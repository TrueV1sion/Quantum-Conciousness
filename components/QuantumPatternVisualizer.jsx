import React, { useState, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, Heatmap } from 'recharts';
import { Card, CardHeader, CardContent } from '@/components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { processQuantumStates, formatCorrelationData, calculateQuantumStateStats } from '@/utils/quantum_visualization_utils';

const QuantumPatternVisualizer = ({ 
  patterns, 
  correlations, 
  quantum_relationships,
  onPatternSelect 
}) => {
  const [selectedPattern, setSelectedPattern] = useState(null);
  const [activeTab, setActiveTab] = useState('patterns');
  const [quantumStates, setQuantumStates] = useState([]);
  const [selectedCorrelation, setSelectedCorrelation] = useState(null);

  useEffect(() => {
    if (patterns && patterns.length > 0) {
      const states = processQuantumStates(patterns);
      setQuantumStates(states);
    }
  }, [patterns]);

  const handlePatternClick = (pattern) => {
    setSelectedPattern(pattern);
    if (onPatternSelect) {
      onPatternSelect(pattern);
    }
  };

  const handleCorrelationClick = (correlation) => {
    setSelectedCorrelation(correlation);
  };

  return (
    <div className="w-full space-y-4">
      <Tabs value={activeTab} onValueChange={setActiveTab}>
        <TabsList>
          <TabsTrigger value="patterns">Patterns</TabsTrigger>
          <TabsTrigger value="correlations">Correlations</TabsTrigger>
          <TabsTrigger value="quantum">Quantum States</TabsTrigger>
          <TabsTrigger value="relationships">Quantum Relationships</TabsTrigger>
        </TabsList>

        <TabsContent value="patterns">
          <Card>
            <CardHeader>
              <h3 className="text-lg font-semibold">Pattern Analysis</h3>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {patterns.map((pattern, index) => (
                  <div
                    key={index}
                    className={`p-4 border rounded hover:bg-gray-50 cursor-pointer ${
                      selectedPattern?.id === pattern.id ? 'border-primary' : ''
                    }`}
                    onClick={() => handlePatternClick(pattern)}
                  >
                    <div className="flex justify-between items-start">
                      <div>
                        <h4 className="font-medium">{pattern.modality} Pattern</h4>
                        <p className="text-sm text-gray-600">
                          Type: {pattern.pattern_type}
                        </p>
                      </div>
                      <Badge variant={pattern.strength > 0.7 ? 'default' : 'secondary'}>
                        {pattern.strength.toFixed(2)}
                      </Badge>
                    </div>
                    {pattern.temporal_span && (
                      <p className="text-sm text-gray-600 mt-2">
                        Time Range: {pattern.temporal_span.join(' to ')}
                      </p>
                    )}
                    {pattern.quantum_metrics && (
                      <div className="mt-2 text-sm">
                        <p>Quantum Coherence: {pattern.quantum_metrics.coherence?.toFixed(3)}</p>
                        <p>Entanglement: {pattern.quantum_metrics.entanglement?.toFixed(3)}</p>
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="correlations">
          <Card>
            <CardHeader>
              <h3 className="text-lg font-semibold">Correlation Analysis</h3>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
                <div className="space-y-4">
                  {correlations.map((correlation, index) => (
                    <div
                      key={index}
                      className={`p-4 border rounded hover:bg-gray-50 cursor-pointer ${
                        selectedCorrelation === correlation ? 'border-primary' : ''
                      }`}
                      onClick={() => handleCorrelationClick(correlation)}
                    >
                      <div className="flex justify-between items-start">
                        <div>
                          <h4 className="font-medium">
                            {correlation.pattern_type} Correlation
                          </h4>
                          <p className="text-sm text-gray-600">
                            Strength: {correlation.strength.toFixed(3)}
                          </p>
                        </div>
                        <Badge variant={correlation.confidence > 0.8 ? 'default' : 'secondary'}>
                          {(correlation.confidence * 100).toFixed(1)}% Confidence
                        </Badge>
                      </div>
                      <div className="mt-2 text-sm">
                        <p>Temporal: {correlation.temporal_relationship}</p>
                        {correlation.quantum_metrics && (
                          <>
                            <p>
                              Quantum Correlation: {
                                correlation.quantum_metrics.quantum_correlation?.toFixed(3)
                              }
                            </p>
                            <p>
                              Entanglement: {
                                correlation.quantum_metrics.entanglement?.toFixed(3)
                              }
                            </p>
                          </>
                        )}
                      </div>
                    </div>
                  ))}
                </div>
                {selectedCorrelation && (
                  <div className="space-y-4">
                    <Card>
                      <CardHeader>
                        <h4 className="font-medium">Correlation Evidence</h4>
                      </CardHeader>
                      <CardContent>
                        {selectedCorrelation.evidence.map((evidence, index) => (
                          <div key={index} className="mb-4">
                            <h5 className="font-medium capitalize">
                              {evidence.type} Evidence
                            </h5>
                            <div className="mt-1 text-sm">
                              {Object.entries(evidence.details).map(([key, value]) => (
                                <p key={key}>
                                  {key.replace(/_/g, ' ')}: {
                                    typeof value === 'number' ? value.toFixed(3) : value
                                  }
                                </p>
                              ))}
                            </div>
                          </div>
                        ))}
                      </CardContent>
                    </Card>
                  </div>
                )}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="quantum">
          <Card>
            <CardHeader>
              <h3 className="text-lg font-semibold">Quantum State Analysis</h3>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
                <div className="h-96">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={quantumStates[0]?.stateComponents || []}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="index" label="State Component" />
                      <YAxis label="Amplitude" />
                      <Tooltip />
                      <Legend />
                      {quantumStates.map((state, index) => (
                        <Line
                          key={index}
                          type="monotone"
                          dataKey="amplitude"
                          data={state.stateComponents}
                          name={`${state.modality} Pattern`}
                          stroke={`hsl(${index * 360 / quantumStates.length}, 70%, 50%)`}
                        />
                      ))}
                    </LineChart>
                  </ResponsiveContainer>
                </div>
                <div>
                  <h4 className="font-medium mb-4">Quantum State Statistics</h4>
                  <div className="space-y-4">
                    {calculateQuantumStateStats(quantumStates).map((stats, index) => (
                      <div key={index} className="p-4 border rounded">
                        <h5 className="font-medium">{stats.modality} Pattern</h5>
                        <div className="mt-2 text-sm grid grid-cols-2 gap-2">
                          <p>Mean: {stats.mean.toFixed(3)}</p>
                          <p>Variance: {stats.variance.toFixed(3)}</p>
                          <p>Max: {stats.max.toFixed(3)}</p>
                          <p>Min: {stats.min.toFixed(3)}</p>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="relationships">
          <Card>
            <CardHeader>
              <h3 className="text-lg font-semibold">Quantum Relationships</h3>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {quantum_relationships.map((relationship, index) => (
                  <div key={index} className="p-4 border rounded">
                    <div className="flex justify-between items-start">
                      <h4 className="font-medium capitalize">
                        {relationship.type.replace(/_/g, ' ')}
                      </h4>
                      {relationship.cluster_id !== undefined && (
                        <Badge>Cluster {relationship.cluster_id}</Badge>
                      )}
                    </div>
                    <div className="mt-2 text-sm">
                      {relationship.metrics && Object.entries(relationship.metrics).map(
                        ([key, value]) => (
                          <p key={key}>
                            {key.replace(/_/g, ' ')}: {
                              typeof value === 'number' ? value.toFixed(3) : value
                            }
                          </p>
                        )
                      )}
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>

      {selectedPattern && (
        <Card>
          <CardHeader>
            <h3 className="text-lg font-semibold">Pattern Details</h3>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <h4 className="font-medium">Basic Information</h4>
                  <p>Modality: {selectedPattern.modality}</p>
                  <p>Type: {selectedPattern.pattern_type}</p>
                  <p>Strength: {selectedPattern.strength.toFixed(3)}</p>
                </div>
                <div>
                  <h4 className="font-medium">Quantum Metrics</h4>
                  {selectedPattern.quantum_metrics && Object.entries(selectedPattern.quantum_metrics).map(([key, value]) => (
                    <p key={key}>{key}: {value.toFixed(3)}</p>
                  ))}
                </div>
              </div>
              {selectedPattern.correlations && (
                <div>
                  <h4 className="font-medium">Related Patterns</h4>
                  <div className="grid grid-cols-2 gap-2 mt-2">
                    {Object.entries(selectedPattern.correlations).map(([patternId, correlation]) => (
                      <div key={patternId} className="p-2 border rounded">
                        <p>Pattern ID: {patternId}</p>
                        <p>Correlation: {correlation.toFixed(3)}</p>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
};

export default QuantumPatternVisualizer; 