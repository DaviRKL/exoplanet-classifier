import { Component, OnInit } from '@angular/core';
import { trigger, state, style, animate, transition } from '@angular/animations';
import { Chart, ChartConfiguration, ChartData, registerables } from 'chart.js';
import { timer } from 'rxjs';

// Register all Chart.js components to prevent errors
Chart.register(...registerables);

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

  // --- Mock Data for the UI (in English) ---
  lastPredictedPlanets = [
    'Kepler-186f', 'TRAPPIST-1e', 'Proxima Centauri b', 'Kepler-452b',
    'Gliese 581g', 'HD 209458 b', '55 Cancri e', 'Kepler-22b',
    'LHS 1140 b', 'CoRoT-7b'
  ];
  
  // New list for the advanced mode
  latestExoplanetDiscoveries = [
    { name: 'Kepler-186f', status: 'CONFIRMED', radius: '1.17 R⊕', discovery: '2014' },
    { name: 'TRAPPIST-1e', status: 'CONFIRMED', radius: '0.91 R⊕', discovery: '2017' },
    { name: 'Proxima Centauri b', status: 'CONFIRMED', radius: '1.1 R⊕', discovery: '2016' },
    { name: 'Kepler-452b', status: 'CONFIRMED', radius: '1.63 R⊕', discovery: '2015' },
    { name: 'Gliese 581g', status: 'UNCONFIRMED', radius: '1.5 R⊕', discovery: '2010' },
    { name: 'HD 209458 b', status: 'CONFIRMED', radius: '1.38 Rj', discovery: '1999' },
    { name: '55 Cancri e', status: 'CONFIRMED', radius: '1.87 R⊕', discovery: '2004' },
    { name: 'Kepler-22b', status: 'CONFIRMED', radius: '2.4 R⊕', discovery: '2011' },
    { name: 'LHS 1140 b', status: 'CONFIRMED', radius: '1.73 R⊕', discovery: '2017' },
    { name: 'CoRoT-7b', status: 'CONFIRMED', radius: '1.58 R⊕', discovery: '2009' },
  ];
  
  bestModelName: string = 'Random Forest';
  confusionMatrix: number[][] = [
    [1250, 50],
    [80, 1420]
  ];
  confusionMatrixClasses: string[] = ['NEGATIVE', 'POSITIVE'];

  // --- Chart Configurations ---
  public lineChartData: ChartConfiguration['data'] | null = null;
  public pieChartData: ChartData<'pie'> | null = null;
  public barChartData: ChartData<'bar'> | null = null;

  public commonChartOptions: ChartConfiguration['options'] = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        labels: {
          color: 'var(--text-primary)',
          font: { size: 12 }
        }
      },
    },
    scales: {
      x: { ticks: { color: 'var(--text-secondary)' }, grid: { color: 'rgba(255, 255, 255, 0.1)' } },
      y: { ticks: { color: 'var(--text-secondary)' }, grid: { color: 'rgba(255, 255, 255, 0.1)' } }
    }
  };

  public pieChartOptions: ChartConfiguration['options'] = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        display: true,
        position: 'top',
        labels: {
          color: 'var(--text-primary)',
          font: { size: 12 }
        }
      }
    }
  };

  constructor() {}

  ngOnInit(): void {
    this.refreshData();
  }

  refreshData(): void {
    this.loading = true;
    this.error = null;
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
      datasets: [{
        data: [95.8, 4.2],
        backgroundColor: ['#4ade80', '#f87171'],
        borderColor: 'var(--secondary-bg)',
      }]
    };
  }
  
  private prepareBarChart(): void {
    this.barChartData = {
      labels: ['koi_fpflag_ss', 'koi_fpflag_co', 'koi_duration_err1', 'koi_prad', 'koi_steff_err1'],
      datasets: [{
        label: 'Importance',
        data: [0.18, 0.15, 0.12, 0.09, 0.07],
        backgroundColor: 'rgba(226, 139, 18, 0.7)',
        borderColor: 'var(--accent-color)',
        borderWidth: 1,
      }]
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