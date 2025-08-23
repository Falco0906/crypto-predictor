import React, { useState, useEffect, useCallback } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { TrendingUp, TrendingDown, Brain, Zap, AlertCircle, RefreshCw, Activity } from 'lucide-react';
import * as tf from 'tensorflow';

const CryptoWebPredictor = () => {
  const [model, setModel] = useState(null);
  const [modelMetadata, setModelMetadata] = useState(null);
  const [predictions, setPredictions] = useState({});
  const [liveData, setLiveData] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [lastUpdate, setLastUpdate] = useState(null);
  const [modelStatus, setModelStatus] = useState('loading');

  // Mock real-time crypto data (in production, you'd connect to an API)
  const generateMockCryptoData = useCallback((basePrice, volatility = 0.02) => {
    const now = Date.now();
    return {
      timestamp: now,
      price: basePrice * (1 + (Math.random() - 0.5) * volatility),
      volume: Math.random() * 1000000,
      change24h: (Math.random() - 0.5) * 10,
    };
  }, []);

  // Load pre-trained model (simulated - in production you'd load from your server)
  const loadPretrainedModel = useCallback(async () => {
    try {
      setModelStatus('loading');
      
      // Simulate loading delay
      await new Promise(resolve => setTimeout(resolve, 2000));
      
      // In a real implementation, you would:
      // const model = await tf.loadLayersModel('/path/to/your/trained_model/model.json');
      // const metadata = await fetch('/path/to/model_metadata.json').then(r => r.json());
      
      // For demo purposes, we'll create a mock model structure
      const mockMetadata = {
        sequence_length: 60,
        prediction_days: 1,
        feature_columns: [
          'close', 'sma_7', 'sma_21', 'sma_50', 'ema_12', 'ema_26',
          'macd', 'macd_signal', 'rsi', 'bb_width', 'bb_position',
          'price_change', 'volatility', 'volume_ratio'
        ],
        target_column: 'close',
        model_type: 'LSTM',
        training_date: '2024-01-15',
        accuracy: 87.3
      };

      // Create a simple mock model for demonstration
      const mockModel = {
        predict: (inputData) => {
          // Simulate LSTM prediction logic
          const lastPrice = inputData[inputData.length - 1][0];
          const trend = Math.random() > 0.5 ? 1.02 : 0.98; // Random trend
          const noise = (Math.random() - 0.5) * 0.01;
          return [[lastPrice * trend + noise]];
        }
      };

      setModel(mockModel);
      setModelMetadata(mockMetadata);
      setModelStatus('loaded');
      setLoading(false);
      
    } catch (err) {
      console.error('Error loading model:', err);
      setError('Failed to load pre-trained model');
      setModelStatus('error');
      setLoading(false);
    }
  }, []);

  // Calculate technical indicators (simplified versions)
  const calculateIndicators = useCallback((priceData) => {
    if (priceData.length < 20) return null;

    const prices = priceData.map(d => d.price);
    const latest = priceData[priceData.length - 1];

    // Simple Moving Averages
    const sma_7 = prices.slice(-7).reduce((a, b) => a + b, 0) / 7;
    const sma_21 = prices.slice(-21).reduce((a, b) => a + b, 0) / Math.min(21, prices.length);
    
    // RSI calculation (simplified)
    const changes = [];
    for (let i = 1; i < Math.min(15, prices.length); i++) {
      changes.push(prices[prices.length - i] - prices[prices.length - i - 1]);
    }
    const gains = changes.filter(c => c > 0).reduce((a, b) => a + b, 0) / changes.length;
    const losses = Math.abs(changes.filter(c => c < 0).reduce((a, b) => a + b, 0)) / changes.length;
    const rsi = losses === 0 ? 100 : 100 - (100 / (1 + gains / losses));

    // Price change
    const priceChange = prices.length > 1 ? 
      ((prices[prices.length - 1] - prices[prices.length - 2]) / prices[prices.length - 2]) * 100 : 0;

    return {
      price: latest.price,
      sma_7,
      sma_21,
      rsi,
      priceChange,
      volume: latest.volume,
      timestamp: latest.timestamp
    };
  }, []);

  // Make prediction using loaded model
  const makePrediction = useCallback((coin, historicalData) => {
    if (!model || !modelMetadata || historicalData.length < modelMetadata.sequence_length) {
      return null;
    }

    try {
      // Prepare input data (this would use your actual feature engineering logic)
      const indicators = calculateIndicators(historicalData);
      if (!indicators) return null;

      // Create mock input sequence (in production, use your actual preprocessing)
      const inputSequence = [];
      for (let i = 0; i < modelMetadata.sequence_length; i++) {
        inputSequence.push([
          indicators.price,
          indicators.sma_7,
          indicators.sma_21,
          indicators.rsi / 100, // Normalize
          indicators.priceChange / 100, // Normalize
          Math.random() * 0.1, // Mock other features
        ]);
      }

      // Make prediction
      const prediction = model.predict([inputSequence]);
      const predictedPrice = prediction[0][0];
      
      const currentPrice = indicators.price;
      const priceChange = ((predictedPrice - currentPrice) / currentPrice) * 100;
      const confidence = Math.random() * 20 + 70; // Mock confidence score

      return {
        currentPrice,
        predictedPrice,
        priceChange,
        confidence,
        timestamp: Date.now(),
        indicators
      };
    } catch (err) {
      console.error('Prediction error:', err);
      return null;
    }
  }, [model, modelMetadata, calculateIndicators]);

  // Simulate real-time data updates
  useEffect(() => {
    if (modelStatus !== 'loaded') return;

    const cryptos = {
      'Bitcoin': { price: 45000, volatility: 0.03 },
      'Ethereum': { price: 3200, volatility: 0.04 },
      'Solana': { price: 180, volatility: 0.06 },
      'Litecoin': { price: 150, volatility: 0.05 }
    };

    const updateData = () => {
      const newLiveData = [];
      const newPredictions = {};

      Object.entries(cryptos).forEach(([coin, config]) => {
        // Generate new price data
        const newDataPoint = generateMockCryptoData(config.price, config.volatility);
        newLiveData.push({ coin, ...newDataPoint });

        // Update historical data (keep last 100 points)
        const existingData = liveData.filter(d => d.coin === coin).slice(-99);
        const fullHistory = [...existingData, newDataPoint];

        // Make prediction
        const prediction = makePrediction(coin, fullHistory);
        if (prediction) {
          newPredictions[coin] = prediction;
        }
      });

      setLiveData(prev => {
        const combined = [...prev, ...newLiveData];
        return combined.slice(-400); // Keep last 400 data points
      });

      setPredictions(prev => ({ ...prev, ...newPredictions }));
      setLastUpdate(Date.now());
    };

    // Initial update
    updateData();

    // Set up interval for updates
    const interval = setInterval(updateData, 5000); // Update every 5 seconds

    return () => clearInterval(interval);
  }, [modelStatus, generateMockCryptoData, makePrediction]);

  // Load model on component mount
  useEffect(() => {
    loadPretrainedModel();
  }, [loadPretrainedModel]);

  // Get chart data for a specific coin
  const getChartData = (coin) => {
    return liveData
      .filter(d => d.coin === coin)
      .slice(-50) // Last 50 data points
      .map((d, i) => ({
        time: i,
        price: d.price.toFixed(2),
        timestamp: d.timestamp
      }));
  };

  // Format price with appropriate decimals
  const formatPrice = (price) => {
    if (price > 1000) return `$${price.toFixed(0).toLocaleString()}`;
    if (price > 1) return `$${price.toFixed(2)}`;
    return `$${price.toFixed(4)}`;
  };

  // Format percentage change
  const formatChange = (change) => {
    const sign = change > 0 ? '+' : '';
    return `${sign}${change.toFixed(2)}%`;
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 text-white flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin w-16 h-16 border-4 border-purple-500 border-t-transparent rounded-full mx-auto mb-6"></div>
          <h2 className="text-2xl font-bold mb-2">Loading AI Model</h2>
          <p className="text-gray-300">Initializing pre-trained LSTM neural network...</p>
          <div className="mt-4 text-sm text-purple-400">
            Status: {modelStatus}
          </div>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-900 via-red-900 to-slate-900 text-white flex items-center justify-center">
        <div className="text-center">
          <AlertCircle className="w-16 h-16 text-red-400 mx-auto mb-4" />
          <h2 className="text-2xl font-bold mb-2">Error Loading Model</h2>
          <p className="text-gray-300 mb-4">{error}</p>
          <button 
            onClick={loadPretrainedModel}
            className="bg-red-600 hover:bg-red-700 px-6 py-2 rounded-full"
          >
            Retry Loading
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 text-white p-6">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="text-center mb-8">
          <div className="flex items-center justify-center gap-3 mb-4">
            <Brain className="w-12 h-12 text-purple-400" />
            <h1 className="text-5xl font-bold bg-gradient-to-r from-purple-400 to-pink-400 bg-clip-text text-transparent">
              Crypto AI Predictor
            </h1>
            <Activity className="w-8 h-8 text-green-400 animate-pulse" />
          </div>
          <p className="text-gray-300 text-lg mb-4">
            Real-time cryptocurrency price predictions using pre-trained LSTM neural network
          </p>
          
          {/* Model Status */}
          <div className="flex items-center justify-center gap-6 text-sm">
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 bg-green-500 rounded-full animate-pulse"></div>
              <span>Model: {modelMetadata?.model_type} Active</span>
            </div>
            <div className="flex items-center gap-2">
              <Zap className="w-4 h-4 text-yellow-400" />
              <span>Accuracy: {modelMetadata?.accuracy}%</span>
            </div>
            <div className="flex items-center gap-2">
              <RefreshCw className="w-4 h-4 text-blue-400" />
              <span>Last Update: {lastUpdate ? new Date(lastUpdate).toLocaleTimeString() : 'Never'}</span>
            </div>
          </div>
        </div>

        {/* Prediction Cards */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
          {Object.entries(predictions).map(([coin, prediction]) => {
            const isPositive = prediction.priceChange > 0;
            const chartData = getChartData(coin);
            
            return (
              <div key={coin} className="bg-gray-800/50 backdrop-blur-sm border border-gray-700 rounded-xl p-6 hover:border-purple-500/50 transition-all duration-300">
                <div className="flex items-center justify-between mb-4">
                  <h3 className="text-xl font-bold text-purple-400">{coin}</h3>
                  {isPositive ? (
                    <TrendingUp className="w-6 h-6 text-green-400" />
                  ) : (
                    <TrendingDown className="w-6 h-6 text-red-400" />
                  )}
                </div>
                
                {/* Current Price */}
                <div className="mb-3">
                  <div className="text-2xl font-bold">
                    {formatPrice(prediction.currentPrice)}
                  </div>
                  <div className="text-sm text-gray-400">Current Price</div>
                </div>
                
                {/* Prediction */}
                <div className="mb-4">
                  <div className="text-lg font-semibold text-yellow-400">
                    {formatPrice(prediction.predictedPrice)}
                  </div>
                  <div className="text-sm text-gray-400">24h Prediction</div>
                </div>
                
                {/* Change */}
                <div className="flex items-center justify-between mb-4">
                  <div className={`text-sm font-medium ${isPositive ? 'text-green-400' : 'text-red-400'}`}>
                    {formatChange(prediction.priceChange)}
                  </div>
                  <div className="text-xs text-gray-500">
                    Confidence: {prediction.confidence.toFixed(1)}%
                  </div>
                </div>
                
                {/* Mini Chart */}
                <div className="h-20">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={chartData}>
                      <Line 
                        type="monotone" 
                        dataKey="price" 
                        stroke={isPositive ? "#10B981" : "#EF4444"} 
                        strokeWidth={2}
                        dot={false}
                      />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
                
                {/* Technical Indicators */}
                <div className="mt-4 pt-4 border-t border-gray-700">
                  <div className="grid grid-cols-2 gap-2 text-xs">
                    <div>
                      <span className="text-gray-400">RSI:</span>
                      <span className={`ml-1 ${prediction.indicators?.rsi > 70 ? 'text-red-400' : prediction.indicators?.rsi < 30 ? 'text-green-400' : 'text-gray-300'}`}>
                        {prediction.indicators?.rsi.toFixed(1)}
                      </span>
                    </div>
                    <div>
                      <span className="text-gray-400">SMA7:</span>
                      <span className="ml-1 text-gray-300">
                        {formatPrice(prediction.indicators?.sma_7)}
                      </span>
                    </div>
                  </div>
                </div>
              </div>
            );
          })}
        </div>

        {/* Detailed Charts */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
          {Object.keys(predictions).slice(0, 2).map(coin => {
            const chartData = getChartData(coin);
            const prediction = predictions[coin];
            
            return (
              <div key={coin} className="bg-gray-800/50 backdrop-blur-sm border border-gray-700 rounded-xl p-6">
                <div className="flex items-center justify-between mb-4">
                  <h3 className="text-xl font-bold">{coin} Price Chart</h3>
                  <div className="text-sm text-gray-400">
                    Live data (5s intervals)
                  </div>
                </div>
                
                <div className="h-64 mb-4">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={chartData}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                      <XAxis 
                        dataKey="time" 
                        stroke="#9CA3AF"
                        tick={{ fontSize: 12 }}
                      />
                      <YAxis 
                        stroke="#9CA3AF"
                        tick={{ fontSize: 12 }}
                        domain={['dataMin - 10', 'dataMax + 10']}
                      />
                      <Tooltip 
                        contentStyle={{ 
                          backgroundColor: '#1F2937', 
                          border: '1px solid #374151',
                          borderRadius: '8px',
                          fontSize: '12px'
                        }}
                        formatter={(value) => [formatPrice(parseFloat(value)), 'Price']}
                        labelFormatter={(label) => `Point ${label}`}
                      />
                      <Line 
                        type="monotone" 
                        dataKey="price" 
                        stroke="#8B5CF6" 
                        strokeWidth={2}
                        dot={{ fill: '#8B5CF6', strokeWidth: 2, r: 3 }}
                        activeDot={{ r: 5, fill: '#A855F7' }}
                      />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
                
                {/* Prediction Summary */}
                <div className="bg-gray-700/50 rounded-lg p-4">
                  <div className="grid grid-cols-3 gap-4 text-center">
                    <div>
                      <div className="text-lg font-bold text-blue-400">
                        {formatPrice(prediction?.currentPrice)}
                      </div>
                      <div className="text-xs text-gray-400">Current</div>
                    </div>
                    <div>
                      <div className="text-lg font-bold text-yellow-400">
                        {formatPrice(prediction?.predictedPrice)}
                      </div>
                      <div className="text-xs text-gray-400">Predicted</div>
                    </div>
                    <div>
                      <div className={`text-lg font-bold ${prediction?.priceChange > 0 ? 'text-green-400' : 'text-red-400'}`}>
                        {formatChange(prediction?.priceChange)}
                      </div>
                      <div className="text-xs text-gray-400">Change</div>
                    </div>
                  </div>
                </div>
              </div>
            );
          })}
        </div>

        {/* Model Information */}
        <div className="bg-gray-800/50 backdrop-blur-sm border border-gray-700 rounded-xl p-6">
          <h3 className="text-xl font-bold mb-4 flex items-center gap-2">
            <Brain className="w-6 h-6 text-purple-400" />
            AI Model Information
          </h3>
          
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
            <div className="text-center">
              <div className="text-2xl font-bold text-purple-400">LSTM</div>
              <div className="text-sm text-gray-400">Neural Network</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-green-400">{modelMetadata?.accuracy}%</div>
              <div className="text-sm text-gray-400">Accuracy</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-blue-400">{modelMetadata?.sequence_length}</div>
              <div className="text-sm text-gray-400">Sequence Length</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-yellow-400">{modelMetadata?.feature_columns?.length}</div>
              <div className="text-sm text-gray-400">Features</div>
            </div>
          </div>
          
          <div className="mt-6 pt-6 border-t border-gray-700">
            <h4 className="font-semibold mb-3">Technical Indicators Used:</h4>
            <div className="flex flex-wrap gap-2">
              {modelMetadata?.feature_columns?.slice(0, 10).map((feature, index) => (
                <span key={index} className="px-3 py-1 bg-purple-900/50 text-purple-300 text-xs rounded-full">
                  {feature.replace('_', ' ').toUpperCase()}
                </span>
              ))}
              {modelMetadata?.feature_columns?.length > 10 && (
                <span className="px-3 py-1 bg-gray-700 text-gray-400 text-xs rounded-full">
                  +{modelMetadata.feature_columns.length - 10} more
                </span>
              )}
            </div>
          </div>
        </div>

        {/* Footer */}
        <div className="mt-8 text-center">
          <div className="flex items-center justify-center gap-4 text-sm text-gray-400">
            <div className="flex items-center gap-2">
              <AlertCircle className="w-4 h-4" />
              <span>This is for educational purposes only</span>
            </div>
            <div>•</div>
            <div>Not financial advice</div>
            <div>•</div>
            <div>Model trained: {modelMetadata?.training_date}</div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default CryptoWebPredictor;