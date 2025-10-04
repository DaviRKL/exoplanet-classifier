import { Component, OnInit } from '@angular/core';
import { ApiService } from '../../services/api.service';
import { PlotResponse, TrainingResponse } from '../../models/api.models';

@Component({
  selector: 'app-metrics',
  templateUrl: './metrics.component.html',
  styleUrls: ['./metrics.component.css']
})
export class MetricsComponent implements OnInit {
  metrics: TrainingResponse | null = null;
  plots: PlotResponse | null = null;
  loading = false;
  error: string | null = null;
  displayedColumns = ['model', 'accuracy', 'precision', 'recall', 'f1'];

  constructor(private api: ApiService) {}

  ngOnInit(): void {
    this.fetchData();
  }

  fetchData(): void {
    this.loading = true;
    this.error = null;

    this.api.getMetrics().subscribe({
      next: (metrics) => {
        this.metrics = metrics;
        this.api.getPlots().subscribe({
          next: (plots) => {
            this.plots = plots;
            this.loading = false;
          },
          error: (err) => {
            this.error = err.message ?? 'Unable to load the plots.';
            this.loading = false;
          }
        });
      },
      error: (err) => {
        this.error = err.message ?? 'Unable to load the metrics.';
        this.loading = false;
      }
    });
  }

  asArray(metrics: Record<string, any>): { model: string; data: any }[] {
    return Object.entries(metrics || {}).map(([model, data]) => ({ model, data }));
  }
}
