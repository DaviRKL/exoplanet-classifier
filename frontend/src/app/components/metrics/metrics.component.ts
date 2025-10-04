import { Component, OnInit } from '@angular/core';
import { ApiService } from '../../services/api.service';
import { TrainingResponse } from '../../models/api.models';
import { ChartConfiguration, ChartData } from 'chart.js';
import { forkJoin } from 'rxjs';

@Component({
  selector: 'app-metrics',
  templateUrl: './metrics.component.html',
  styleUrls: ['./metrics.component.css']
})
export class MetricsComponent implements OnInit {
  metrics: TrainingResponse | null = null;
  loading = false;
  error: string | null = null;
  displayedColumns = ['model', 'accuracy', 'precision', 'recall', 'f1'];

  // Dados para o Gráfico da Matriz de Confusão
  public confusionMatrixChartData: ChartData<'bar'> | null = null;
  public confusionMatrixChartOptions: ChartConfiguration['options'] = {
    responsive: true,
    indexAxis: 'y',
    plugins: {
      legend: { display: true, position: 'top', labels: { color: 'white' } },
      title: { display: true, text: 'Visão Geral da Matriz de Confusão', color: 'white', font: { size: 16 } },
    },
    scales: {
      x: { stacked: true, grid: { color: 'rgba(255, 255, 255, 0.1)' }, ticks: { color: 'white' } },
      y: { stacked: true, grid: { color: 'rgba(255, 255, 255, 0.1)' }, ticks: { color: 'white' } }
    }
  };

  constructor(private api: ApiService) {}

  ngOnInit(): void {
    this.fetchData();
  }

  fetchData(): void {
    this.loading = true;
    this.error = null;
    this.metrics = null;
    this.confusionMatrixChartData = null;

    // Usamos forkJoin para buscar métricas e plots em paralelo
    forkJoin({
      metrics: this.api.getMetrics()
      // Adicionar chamada para plots se a API retornar os dados numéricos
    }).subscribe({
      next: ({ metrics }) => {
        this.metrics = metrics;
        this.prepareChartData(metrics);
        this.loading = false;
      },
      error: (err) => {
        this.error = err.message ?? 'Não foi possível carregar as métricas.';
        this.loading = false;
      }
    });
  }

  prepareChartData(metrics: TrainingResponse): void {
    const bestModelMetrics = metrics.metrics[metrics.best_model];
    if (!bestModelMetrics || !bestModelMetrics.confusion_matrix) {
      return;
    }

    // Exemplo: Transformando uma matriz 3x3 em dados de gráfico
    // Assumindo que a API retorna a matriz [[TN, FP, ..], [FN, TP, ..], ...]
    const matrix = bestModelMetrics.confusion_matrix;
    const classes = bestModelMetrics.classes; // Ex: ['CANDIDATE', 'CONFIRMED', 'FALSE POSITIVE']

    // Simplesmente para demonstração - a lógica exata depende do formato da sua matriz
    const correctlyPredicted = matrix.map((row, i) => row[i]); // Diagonal
    const incorrectlyPredicted = matrix.map((row, i) => row.reduce((a, b) => a + b, 0) - row[i]);

    this.confusionMatrixChartData = {
      labels: classes,
      datasets: [
        { data: correctlyPredicted, label: 'Predições Corretas', backgroundColor: 'rgba(74, 222, 128, 0.7)', stack: 'a' },
        { data: incorrectlyPredicted, label: 'Predições Incorretas', backgroundColor: 'rgba(248, 113, 113, 0.7)', stack: 'a' }
      ]
    };
  }

  asArray(metrics: Record<string, any>): { model: string; data: any }[] {
    return Object.entries(metrics || {}).map(([model, data]) => ({ model, data }));
  }
}