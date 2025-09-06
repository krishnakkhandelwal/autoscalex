# cost_reliability/interactive_dashboard.py
# Flask backend with sliders + Randomize button
# Python 3.13 compatible

from flask import Flask, render_template, jsonify, request
import json
from datetime import datetime
from ml_integration import MLCostIntegration
from traffic_manager import intelligent_traffic_shaping, SystemLoad

app = Flask(__name__)

# Initialize ML integration
ml_integrator = MLCostIntegration()

@app.route('/')
def index():
    return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MUFG Interactive Scaling Dashboard</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        .container { 
            max-width: 1200px; 
            margin: 0 auto; 
            background: rgba(255,255,255,0.95);
            border-radius: 20px;
            padding: 30px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
        }
        .header { 
            text-align: center; 
            background: linear-gradient(135deg, #2c3e50, #34495e);
            color: white; 
            padding: 20px; 
            border-radius: 15px; 
            margin-bottom: 30px;
        }
        .controls-section {
            background: white;
            padding: 25px;
            border-radius: 15px;
            margin-bottom: 20px;
            box-shadow: 0 8px 25px rgba(0,0,0,0.1);
        }
        .controls-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
        }
        .randomize-btn {
            background: linear-gradient(135deg, #e67e22, #d35400);
            color: white;
            border: none;
            padding: 12px 25px;
            border-radius: 25px;
            font-size: 16px;
            font-weight: bold;
            cursor: pointer;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(231, 126, 34, 0.3);
        }
        .randomize-btn:hover {
            background: linear-gradient(135deg, #d35400, #bf3e00);
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(231, 126, 34, 0.4);
        }
        .randomize-btn:active {
            transform: translateY(0);
            animation: pulse 0.3s ease-in-out;
        }
        @keyframes pulse {
            0% { transform: scale(1); }
            50% { transform: scale(0.95); }
            100% { transform: scale(1); }
        }
        .controls-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
        }
        .control-group {
            display: flex;
            flex-direction: column;
            gap: 10px;
        }
        .control-group label {
            font-weight: bold;
            color: #2c3e50;
            font-size: 14px;
        }
        .slider {
            width: 100%;
            height: 8px;
            border-radius: 5px;
            background: #ddd;
            outline: none;
            -webkit-appearance: none;
            transition: background 0.3s ease;
        }
        .slider:hover {
            background: #bbb;
        }
        .slider::-webkit-slider-thumb {
            appearance: none;
            width: 20px;
            height: 20px;
            border-radius: 50%;
            background: #3498db;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        .slider::-webkit-slider-thumb:hover {
            background: #2980b9;
            transform: scale(1.1);
        }
        .slider::-moz-range-thumb {
            width: 20px;
            height: 20px;
            border-radius: 50%;
            background: #3498db;
            cursor: pointer;
            border: none;
        }
        .value-display {
            font-size: 16px;
            font-weight: bold;
            color: #3498db;
            transition: color 0.3s ease;
        }
        .checkbox-group {
            display: flex;
            gap: 20px;
            align-items: center;
            flex-wrap: wrap;
        }
        .checkbox-group input[type="checkbox"] {
            transform: scale(1.5);
            margin-right: 8px;
        }
        .checkbox-group label {
            display: flex;
            align-items: center;
            cursor: pointer;
            transition: color 0.3s ease;
        }
        .checkbox-group label:hover {
            color: #3498db;
        }
        .results-section {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }
        .result-panel {
            background: white;
            padding: 25px;
            border-radius: 15px;
            box-shadow: 0 8px 25px rgba(0,0,0,0.1);
            transition: all 0.3s ease;
        }
        .status-scale {
            background: linear-gradient(135deg, #e74c3c, #c0392b);
            color: white;
            border: 3px solid #e74c3c;
            animation: glow-red 2s ease-in-out infinite alternate;
        }
        .status-no-scale {
            background: linear-gradient(135deg, #27ae60, #2ecc71);
            color: white;
            border: 3px solid #27ae60;
            animation: glow-green 2s ease-in-out infinite alternate;
        }
        .status-unknown {
            background: linear-gradient(135deg, #95a5a6, #7f8c8d);
            color: white;
            border: 3px solid #95a5a6;
        }
        @keyframes glow-red {
            from { box-shadow: 0 0 5px #e74c3c; }
            to { box-shadow: 0 0 20px #e74c3c, 0 0 30px #e74c3c; }
        }
        @keyframes glow-green {
            from { box-shadow: 0 0 5px #27ae60; }
            to { box-shadow: 0 0 20px #27ae60, 0 0 30px #27ae60; }
        }
        .metric-value {
            font-size: 1.8em;
            font-weight: bold;
            margin-bottom: 10px;
        }
        .detail-text {
            font-size: 14px;
            line-height: 1.5;
        }
        .update-indicator {
            position: fixed;
            top: 20px;
            right: 20px;
            background: #3498db;
            color: white;
            padding: 10px 15px;
            border-radius: 20px;
            font-size: 12px;
            transition: all 0.3s ease;
        }
        .randomizing {
            background: #e67e22 !important;
        }
        .randomizing::after {
            content: " 🎲";
        }
    </style>
</head>
<body>
    <div class="update-indicator" id="updateIndicator">
        🔄 Updating...
    </div>
    
    <div class="container">
        <div class="header">
            <h1>🏦 MUFG Interactive Scaling Dashboard</h1>
            <p>Adjust metrics with sliders or randomize - Live recommendations every 2 seconds</p>
        </div>
        
        <div class="controls-section">
            <div class="controls-header">
                <h3 style="color: #2c3e50;">📊 System Metrics Controls</h3>
                <button class="randomize-btn" id="randomizeBtn" onclick="randomizeAll()">
                    🎲 Randomize!!
                </button>
            </div>
            
            <div class="controls-grid">
                <div class="control-group">
                    <label for="activeUsers">Active Users:</label>
                    <input type="range" id="activeUsers" class="slider" min="100" max="5000" value="1000">
                    <div class="value-display" id="activeUsersValue">1000</div>
                </div>
                
                <div class="control-group">
                    <label for="cpuUsage">CPU Usage (%):</label>
                    <input type="range" id="cpuUsage" class="slider" min="10" max="98" value="45">
                    <div class="value-display" id="cpuUsageValue">45%</div>
                </div>
                
                <div class="control-group">
                    <label for="memoryUsage">Memory Usage (%):</label>
                    <input type="range" id="memoryUsage" class="slider" min="10" max="98" value="40">
                    <div class="value-display" id="memoryUsageValue">40%</div>
                </div>
                
                <div class="control-group">
                    <label for="queueDepth">Queue Depth:</label>
                    <input type="range" id="queueDepth" class="slider" min="0" max="500" value="50">
                    <div class="value-display" id="queueDepthValue">50</div>
                </div>
            </div>
            
            <h4 style="margin: 20px 0 10px 0; color: #2c3e50;">🎯 Business Events</h4>
            <div class="checkbox-group">
                <label><input type="checkbox" id="eventPayroll"> Payroll Processing</label>
                <label><input type="checkbox" id="eventTax"> Tax Filing</label>
                <label><input type="checkbox" id="eventEom"> Month-End Reporting</label>
            </div>
        </div>
        
        <div class="results-section">
            <div class="result-panel" id="recommendationPanel">
                <h3>🎯 Scaling Recommendation</h3>
                <div class="metric-value" id="recommendationStatus">Loading...</div>
                <div class="detail-text" id="recommendationDetails">Initializing...</div>
            </div>
            
            <div class="result-panel">
                <h3>📈 System Status</h3>
                <div class="detail-text" id="systemStatus">
                    <strong>Model:</strong> <span id="modelType">Loading...</span><br>
                    <strong>Business Tier:</strong> <span id="businessTier">-</span><br>
                    <strong>SLA Tier:</strong> <span id="slaTier">-</span><br>
                    <strong>Bulkhead State:</strong> <span id="bulkheadState">-</span><br>
                    <strong>Last Update:</strong> <span id="lastUpdate">-</span>
                </div>
            </div>
        </div>
    </div>

    <script>
        let updateInterval;
        
        // Randomize all controls
        function randomizeAll() {
            const btn = document.getElementById('randomizeBtn');
            const indicator = document.getElementById('updateIndicator');
            
            // Visual feedback
            btn.classList.add('randomizing');
            indicator.classList.add('randomizing');
            indicator.textContent = '🎲 Randomizing...';
            
            // Randomize sliders
            const randomValues = {
                activeUsers: Math.floor(Math.random() * (5000 - 100) + 100),
                cpuUsage: Math.floor(Math.random() * (98 - 10) + 10),
                memoryUsage: Math.floor(Math.random() * (98 - 10) + 10),
                queueDepth: Math.floor(Math.random() * 500)
            };
            
            // Apply random values with animation effect
            Object.keys(randomValues).forEach((id, index) => {
                setTimeout(() => {
                    const slider = document.getElementById(id);
                    const display = document.getElementById(id + 'Value');
                    
                    // Animate slider to new position
                    slider.value = randomValues[id];
                    
                    // Update display
                    let value = randomValues[id];
                    if (id.includes('Usage')) {
                        value += '%';
                    }
                    display.textContent = value;
                    
                    // Visual feedback for this slider
                    display.style.color = '#e67e22';
                    setTimeout(() => {
                        display.style.color = '#3498db';
                    }, 500);
                    
                }, index * 100); // Stagger the updates
            });
            
            // Randomize business events (30% chance each)
            setTimeout(() => {
                document.getElementById('eventPayroll').checked = Math.random() < 0.3;
                document.getElementById('eventTax').checked = Math.random() < 0.3;
                document.getElementById('eventEom').checked = Math.random() < 0.3;
                
                // Reset button state
                btn.classList.remove('randomizing');
                indicator.classList.remove('randomizing');
                indicator.textContent = '🔄 Updating...';
                
                // Trigger immediate update
                updateRecommendation();
                
            }, 400);
        }
        
        // Update slider value displays
        function setupSliders() {
            const sliders = ['activeUsers', 'cpuUsage', 'memoryUsage', 'queueDepth'];
            sliders.forEach(id => {
                const slider = document.getElementById(id);
                const display = document.getElementById(id + 'Value');
                
                slider.addEventListener('input', function() {
                    let value = this.value;
                    if (id.includes('Usage')) {
                        value += '%';
                    }
                    display.textContent = value;
                });
            });
        }
        
        // Gather current metric values
        function getCurrentMetrics() {
            return {
                active_users: parseInt(document.getElementById('activeUsers').value),
                cpu_usage: parseFloat(document.getElementById('cpuUsage').value),
                memory_usage: parseFloat(document.getElementById('memoryUsage').value),
                queue_depth: parseInt(document.getElementById('queueDepth').value),
                has_event: document.getElementById('eventPayroll').checked || 
                          document.getElementById('eventTax').checked || 
                          document.getElementById('eventEom').checked ? 1 : 0,
                event_payroll: document.getElementById('eventPayroll').checked ? 1 : 0,
                event_tax: document.getElementById('eventTax').checked ? 1 : 0,
                event_eom: document.getElementById('eventEom').checked ? 1 : 0,
                is_business_hours: new Date().getHours() >= 9 && new Date().getHours() <= 19 ? 1 : 0,
                current_capacity: 1000
            };
        }
        
        // Update recommendation display
        function updateRecommendation() {
            const indicator = document.getElementById('updateIndicator');
            if (!indicator.classList.contains('randomizing')) {
                indicator.style.display = 'block';
            }
            
            const metrics = getCurrentMetrics();
            
            fetch('/api/recommend', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(metrics)
            })
            .then(response => response.json())
            .then(data => {
                const panel = document.getElementById('recommendationPanel');
                const status = document.getElementById('recommendationStatus');
                const details = document.getElementById('recommendationDetails');
                
                if (data.recommendation === 'scale') {
                    panel.className = 'result-panel status-scale';
                    status.textContent = '⚠️ SCALING RECOMMENDED';
                    
                    let detailText = `Method: ${data.top_strategy?.method || 'N/A'}\\n`;
                    detailText += `Cost: ¥${data.top_strategy?.cost_per_hour?.toFixed(2) || '0'}/hour\\n`;
                    detailText += `Reliability: ${(data.top_strategy?.reliability_score * 100)?.toFixed(1) || '0'}%\\n`;
                    detailText += `Probability: ${(data.ml_probability * 100)?.toFixed(1) || '0'}%`;
                    details.innerHTML = detailText.replace(/\\n/g, '<br>');
                } else {
                    panel.className = 'result-panel status-no-scale';
                    status.textContent = '✅ NO SCALING NEEDED';
                    
                    let prob = data.ml_probability || data.probability || 0;
                    details.innerHTML = `Probability: ${(prob * 100).toFixed(1)}%<br>System operating normally`;
                }
                
                // Update system status
                document.getElementById('modelType').textContent = data.model_type || 'unknown';
                document.getElementById('businessTier').textContent = data.business_context?.tier || '-';
                document.getElementById('slaTier').textContent = data.business_context?.sla || '-';
                document.getElementById('bulkheadState').textContent = data.traffic_management?.bulkhead_state || '-';
                document.getElementById('lastUpdate').textContent = new Date().toLocaleTimeString();
                
                if (!indicator.classList.contains('randomizing')) {
                    indicator.style.display = 'none';
                }
            })
            .catch(error => {
                console.error('Error:', error);
                const panel = document.getElementById('recommendationPanel');
                panel.className = 'result-panel status-unknown';
                document.getElementById('recommendationStatus').textContent = '❌ ERROR';
                document.getElementById('recommendationDetails').textContent = 'Failed to get recommendation';
                if (!indicator.classList.contains('randomizing')) {
                    indicator.style.display = 'none';
                }
            });
        }
        
        // Initialize
        setupSliders();
        updateRecommendation();
        
        // Update every 2 seconds
        updateInterval = setInterval(updateRecommendation, 2000);
        
        // Also update when sliders change
        ['activeUsers', 'cpuUsage', 'memoryUsage', 'queueDepth'].forEach(id => {
            document.getElementById(id).addEventListener('input', updateRecommendation);
        });
        ['eventPayroll', 'eventTax', 'eventEom'].forEach(id => {
            document.getElementById(id).addEventListener('change', updateRecommendation);
        });
        
        // Keyboard shortcut for randomize (spacebar)
        document.addEventListener('keydown', function(event) {
            if (event.code === 'Space' && !event.target.matches('input')) {
                event.preventDefault();
                randomizeAll();
            }
        });
    </script>
</body>
</html>
    """

@app.route('/api/recommend', methods=['POST'])
def get_recommendation():
    try:
        metrics = request.json
        
        # Add timestamp and other required fields
        metrics['timestamp'] = datetime.now().isoformat()
        metrics['event_risk_score'] = 0.7 if metrics.get('has_event') else 0.0
        
        # Get ML recommendation
        recommendation = ml_integrator.make_scaling_recommendation(metrics)
        
        # Get traffic management info
        system_load = SystemLoad(
            cpu_usage=metrics['cpu_usage'] / 100.0,
            memory_usage=metrics['memory_usage'] / 100.0,
            active_users=metrics['active_users'],
            queue_depth=metrics['queue_depth'],
            timestamp=datetime.now()
        )
        traffic_config = intelligent_traffic_shaping(system_load, [])
        
        # Add traffic management to response
        recommendation['traffic_management'] = traffic_config
        
        return jsonify(recommendation)
        
    except Exception as e:
        return jsonify({
            'recommendation': 'error',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/api/status')
def get_status():
    return jsonify({
        'status': 'running',
        'model_available': not getattr(ml_integrator, 'is_mock', True),
        'timestamp': datetime.now().isoformat()
    })

def main():
    print("🚀 Starting Interactive MUFG Dashboard with Randomize...")
    print("   Dashboard URL: http://localhost:5002")
    print("   🎲 Click 'Randomize!!' or press SPACEBAR to randomize all metrics!")
    
    app.run(debug=False, host='0.0.0.0', port=5002, use_reloader=False)

if __name__ == "__main__":
    main()
