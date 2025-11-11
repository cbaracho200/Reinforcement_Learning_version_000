"""
Dashboard de monitoramento em tempo real.
"""

import time
import threading
from flask import Flask, render_template, jsonify
from flask_cors import CORS
import plotly.graph_objs as go
import plotly.utils
import json
from collections import deque
from typing import Dict, Any, List
import numpy as np


class TrainingDashboard:
    """Dashboard web para monitoramento de treinamento."""
    
    def __init__(self, port: int = 8080, update_interval: float = 1.0):
        """
        Args:
            port: Porta para servir dashboard
            update_interval: Intervalo de atualização em segundos
        """
        
        self.port = port
        self.update_interval = update_interval
        
        # Flask app
        self.app = Flask(__name__)
        CORS(self.app)
        
        # Buffers de dados
        self.metrics_buffer = deque(maxlen=1000)
        self.episode_buffer = deque(maxlen=100)
        self.current_metrics = {}
        
        # Thread seguro
        self.lock = threading.Lock()
        
        # Status
        self.is_running = False
        self.start_time = time.time()
        
        # Configura rotas
        self._setup_routes()
    
    def _setup_routes(self):
        """Configura rotas do Flask."""
        
        @self.app.route('/')
        def index():
            return render_template('dashboard.html')
        
        @self.app.route('/api/metrics')
        def get_metrics():
            with self.lock:
                return jsonify(self.current_metrics)
        
        @self.app.route('/api/history')
        def get_history():
            with self.lock:
                return jsonify({
                    'metrics': list(self.metrics_buffer),
                    'episodes': list(self.episode_buffer)
                })
        
        @self.app.route('/api/plots/<plot_type>')
        def get_plot(plot_type):
            if plot_type == 'rewards':
                return self._create_rewards_plot()
            elif plot_type == 'losses':
                return self._create_losses_plot()
            elif plot_type == 'constraints':
                return self._create_constraints_plot()
            elif plot_type == 'risk':
                return self._create_risk_plot()
            else:
                return jsonify({'error': 'Unknown plot type'})
    
    def update_metrics(self, metrics: Dict[str, Any]):
        """Atualiza métricas atuais."""
        with self.lock:
            self.current_metrics.update(metrics)
            self.current_metrics['timestamp'] = time.time()
            self.metrics_buffer.append(self.current_metrics.copy())
    
    def add_episode(self, episode_data: Dict[str, Any]):
        """Adiciona dados de episódio."""
        with self.lock:
            episode_data['timestamp'] = time.time()
            self.episode_buffer.append(episode_data)
    
    def _create_rewards_plot(self) -> str:
        """Cria gráfico de recompensas."""
        with self.lock:
            if not self.episode_buffer:
                return jsonify({})
            
            episodes = list(self.episode_buffer)
        
        # Extrai dados
        x = [ep.get('episode', i) for i, ep in enumerate(episodes)]
        y = [ep.get('reward', 0) for ep in episodes]
        
        # Média móvel
        window = min(10, len(y))
        if window > 1:
            y_smooth = np.convolve(y, np.ones(window)/window, mode='valid')
            x_smooth = x[window-1:]
        else:
            y_smooth = y
            x_smooth = x
        
        # Cria gráfico
        fig = go.Figure()
        
        # Valores brutos
        fig.add_trace(go.Scatter(
            x=x, y=y,
            mode='markers',
            name='Episodes',
            marker=dict(size=6, opacity=0.5)
        ))
        
        # Média móvel
        fig.add_trace(go.Scatter(
            x=x_smooth, y=y_smooth,
            mode='lines',
            name=f'Média ({window} eps)',
            line=dict(width=2)
        ))
        
        fig.update_layout(
            title='Recompensa por Episódio',
            xaxis_title='Episódio',
            yaxis_title='Recompensa',
            template='plotly_dark'
        )
        
        return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    
    def _create_losses_plot(self) -> str:
        """Cria gráfico de perdas."""
        with self.lock:
            if not self.metrics_buffer:
                return jsonify({})
            
            metrics = list(self.metrics_buffer)
        
        # Extrai perdas
        x = list(range(len(metrics)))
        policy_loss = [m.get('policy_loss', 0) for m in metrics]
        value_loss = [m.get('value_loss', 0) for m in metrics]
        
        # Cria gráfico
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=x, y=policy_loss,
            mode='lines',
            name='Policy Loss',
            line=dict(width=2)
        ))
        
        fig.add_trace(go.Scatter(
            x=x, y=value_loss,
            mode='lines',
            name='Value Loss',
            line=dict(width=2)
        ))
        
        fig.update_layout(
            title='Perdas do Treinamento',
            xaxis_title='Iteração',
            yaxis_title='Loss',
            template='plotly_dark'
        )
        
        return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    
    def _create_constraints_plot(self) -> str:
        """Cria gráfico de violações de restrições."""
        with self.lock:
            if not self.metrics_buffer:
                return jsonify({})
            
            metrics = list(self.metrics_buffer)
        
        # Encontra todas as restrições
        constraint_names = set()
        for m in metrics:
            for key in m:
                if 'constraint' in key and 'violation' in key:
                    name = key.split('/')[1]
                    constraint_names.add(name)
        
        # Cria gráfico
        fig = go.Figure()
        
        x = list(range(len(metrics)))
        
        for name in constraint_names:
            key = f'constraint/{name}/violation_mean'
            y = [m.get(key, 0) for m in metrics]
            
            fig.add_trace(go.Scatter(
                x=x, y=y,
                mode='lines',
                name=name,
                line=dict(width=2)
            ))
        
        fig.update_layout(
            title='Violações de Restrições',
            xaxis_title='Iteração',
            yaxis_title='Violação Média',
            template='plotly_dark'
        )
        
        return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    
    def _create_risk_plot(self) -> str:
        """Cria gráfico de métricas de risco."""
        with self.lock:
            if not self.metrics_buffer:
                return jsonify({})
            
            metrics = list(self.metrics_buffer)
        
        x = list(range(len(metrics)))
        cvar = [m.get('cvar', 0) for m in metrics]
        
        # Cria gráfico
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=x, y=cvar,
            mode='lines',
            name='CVaR (10%)',
            line=dict(width=2)
        ))
        
        fig.update_layout(
            title='Conditional Value at Risk',
            xaxis_title='Iteração',
            yaxis_title='CVaR',
            template='plotly_dark'
        )
        
        return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    
    def start(self, threaded: bool = True):
        """Inicia o dashboard."""
        self.is_running = True
        
        if threaded:
            thread = threading.Thread(
                target=lambda: self.app.run(
                    port=self.port,
                    debug=False,
                    use_reloader=False
                )
            )
            thread.daemon = True
            thread.start()
            print(f"Dashboard rodando em http://localhost:{self.port}")
        else:
            self.app.run(port=self.port)
    
    def stop(self):
        """Para o dashboard."""
        self.is_running = False


# Template HTML para o dashboard
DASHBOARD_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Business-RL Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {
            font-family: Arial, sans-serif;
            background: #1a1a1a;
            color: #fff;
            margin: 0;
            padding: 20px;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
        }
        .header {
            text-align: center;
            margin-bottom: 30px;
        }
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        .metric-card {
            background: #2a2a2a;
            border-radius: 10px;
            padding: 20px;
            text-align: center;
        }
        .metric-value {
            font-size: 2em;
            font-weight: bold;
            color: #4CAF50;
        }
        .metric-label {
            color: #999;
            margin-top: 5px;
        }
        .plot-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(600px, 1fr));
            gap: 20px;
        }
        .plot-container {
            background: #2a2a2a;
            border-radius: 10px;
            padding: 10px;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Business-RL Training Dashboard</h1>
        </div>
        
        <div class="metrics-grid" id="metrics">
            <!-- Métricas serão inseridas aqui -->
        </div>
        
        <div class="plot-grid">
            <div class="plot-container">
                <div id="rewards-plot"></div>
            </div>
            <div class="plot-container">
                <div id="losses-plot"></div>
            </div>
            <div class="plot-container">
                <div id="constraints-plot"></div>
            </div>
            <div class="plot-container">
                <div id="risk-plot"></div>
            </div>
        </div>
    </div>
    
    <script>
        // Atualiza métricas
        function updateMetrics() {
            fetch('/api/metrics')
                .then(response => response.json())
                .then(data => {
                    const container = document.getElementById('metrics');
                    container.innerHTML = '';
                    
                    const mainMetrics = [
                        'total_steps', 'total_episodes', 
                        'mean_reward', 'best_reward'
                    ];
                    
                    for (let metric of mainMetrics) {
                        if (data[metric] !== undefined) {
                            const card = document.createElement('div');
                            card.className = 'metric-card';
                            card.innerHTML = `
                                <div class="metric-value">${data[metric].toFixed(2)}</div>
                                <div class="metric-label">${metric.replace('_', ' ')}</div>
                            `;
                            container.appendChild(card);
                        }
                    }
                });
        }
        
        // Atualiza gráficos
        function updatePlots() {
            const plots = ['rewards', 'losses', 'constraints', 'risk'];
            
            for (let plot of plots) {
                fetch(`/api/plots/${plot}`)
                    .then(response => response.json())
                    .then(fig => {
                        if (fig && Object.keys(fig).length > 0) {
                            Plotly.newPlot(`${plot}-plot`, fig);
                        }
                    });
            }
        }
        
        // Atualiza periodicamente
        setInterval(updateMetrics, 1000);
        setInterval(updatePlots, 5000);
        
        // Carrega inicial
        updateMetrics();
        updatePlots();
    </script>
</body>
</html>
"""

# Salva template
import os
os.makedirs('templates', exist_ok=True)
with open('templates/dashboard.html', 'w') as f:
    f.write(DASHBOARD_HTML)