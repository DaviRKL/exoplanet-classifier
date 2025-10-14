import { Component, OnInit } from '@angular/core';
import { trigger, state, style, animate, transition } from '@angular/animations';
import { Chart, ChartConfiguration, ChartData, registerables } from 'chart.js';
import { timer } from 'rxjs';
import { ExoplanetService } from '../../services/exoplanet.service';

Chart.register(...registerables);
// Global Chart.js defaults for better readability on dark theme
// Do NOT overwrite plugins object; extend existing to avoid side effects
// (Chart.defaults as any).color = '#ffffff';
// (Chart.defaults.plugins as any) = (Chart.defaults.plugins as any) || {};
// (Chart.defaults.plugins as any).legend = {
//   ...((Chart.defaults.plugins as any).legend || {}),
//   labels: { color: '#ffffff' }
// } as any;
// (Chart.defaults.plugins as any).title = {
//   ...((Chart.defaults.plugins as any).title || {}),
//   color: '#ffffff'
// } as any;
// const _scales: any = (Chart.defaults as any).scales || {};
// if (_scales.category) {
//   _scales.category.ticks = { ...(_scales.category.ticks || {}), color: '#ffffff' };
//   _scales.category.grid = { ...(_scales.category.grid || {}), color: 'rgba(255,255,255,0.12)' };
// }
// if (_scales.linear) {
//   _scales.linear.ticks = { ...(_scales.linear.ticks || {}), color: '#ffffff' };
//   _scales.linear.grid = { ...(_scales.linear.grid || {}), color: 'rgba(255,255,255,0.12)' };
// }

@Component({
  selector: 'app-dashboard',
  templateUrl: './dashboard.component.html',
  styleUrls: ['./dashboard.component.css'],
  animations: [
    trigger('modeTransition', [
      transition(':enter', [
        style({ opacity: 0, transform: 'translateY(20px)' }),
        animate('400ms ease-out', style({ opacity: 1, transform: 'translateY(0)' })),
      ]),
      transition(':leave', [
        animate('400ms ease-in', style({ opacity: 0, transform: 'translateY(-20px)' }))
      ])
    ])
  ]
})
export class DashboardComponent implements OnInit {
  isAdvancedMode = false;
  loading = true;
  error: string | null = null;

  lastPredictedPlanets: string[] = [];
  latestExoplanetDiscoveries: any[] = [];
  
  bestModelName: string = 'Random Forest';
  confusionMatrix: number[][] = [ [1250, 50], [80, 1420] ];
  confusionMatrixClasses: string[] = ['NEGATIVE', 'POSITIVE'];

  public lineChartData: ChartConfiguration['data'] | null = null;
  public pieChartData: ChartData<'pie'> | null = null;
  public barChartData: ChartData<'bar'> | null = null;

  public commonChartOptions: ChartConfiguration['options'] = {
    responsive: true, maintainAspectRatio: false,
    plugins: { legend: { labels: { color: '#ffffff', font: { size: 12 } } } },
    scales: {
      x: { ticks: { color: '#ffffff' }, grid: { color: 'rgba(255,255,255,0.12)' } },
      y: { ticks: { color: '#ffffff' }, grid: { color: 'rgba(255,255,255,0.12)' } }
    }
  };

  public pieChartOptions: ChartConfiguration['options'] = {
    responsive: true, maintainAspectRatio: false,
    plugins: { legend: { display: true, position: 'top', labels: { color: '#ffffff', font: { size: 12 } } } }
  };

  constructor(private exoplanetService: ExoplanetService) {}

  ngOnInit(): void {
    this.refreshData();
  }

  refreshData(): void {
    this.loading = true;
    this.error = null;
    
    this.exoplanetService.getExoplanetNames().subscribe(names => {
      this.lastPredictedPlanets = names;
    });

    this.exoplanetService.getAllExoplanets().subscribe(details => {
      this.latestExoplanetDiscoveries = details.map(planet => ({
        ...planet,
        status: planet.type.includes('Unconfirmed') ? 'UNCONFIRMED' : 'CONFIRMED'
      }));
    });

    timer(800).subscribe(() => {
      this.prepareAllCharts();
      this.loading = false;
    });
  }
  
  private prepareAllCharts(): void {
    this.preparePieChart();
    this.prepareBarChart();
    this.prepareLineChart();
  }
  
  private preparePieChart(): void {
    this.pieChartData = {
      labels: ['Correct Predictions (%)', 'Incorrect Predictions (%)'],
      datasets: [{ data: [95.8, 4.2], backgroundColor: ['#4ade80', '#f87171'], borderColor: 'var(--secondary-bg)' }]
    };
  }
  
  private prepareBarChart(): void {
    this.barChartData = {
      labels: ['koi_fpflag_ss', 'koi_fpflag_co', 'koi_duration_err1', 'koi_prad', 'koi_steff_err1'],
      datasets: [{ label: 'Importance', data: [0.18, 0.15, 0.12, 0.09, 0.07], backgroundColor: 'rgba(226, 139, 18, 0.7)', borderColor: 'var(--accent-color)', borderWidth: 1 }]
    };
  }

  private prepareLineChart(): void {
    this.lineChartData = {
      labels: ['Epoch 1', 'Epoch 5', 'Epoch 10', 'Epoch 15', 'Epoch 20'],
      datasets: [
        { data: [0.95, 0.96, 0.97, 0.98, 0.985], label: 'Accuracy', tension: 0.4, borderColor: '#4ade80', pointBackgroundColor: '#4ade80' },
        { data: [0.94, 0.95, 0.96, 0.97, 0.98], label: 'Precision', tension: 0.4, borderColor: '#60a5fa', pointBackgroundColor: '#60a5fa' },
        { data: [0.35, 0.21, 0.15, 0.11, 0.09], label: 'Loss', tension: 0.4, borderColor: '#f87171', pointBackgroundColor: '#f87171' },
      ]
    };
  }
}
