import { Component, OnInit } from '@angular/core';
import { ApiService } from '../../services/api.service';
import { HealthResponse } from '../../models/api.models';

@Component({
  selector: 'app-dashboard',
  templateUrl: './dashboard.component.html',
  styleUrls: ['./dashboard.component.css']
})
export class DashboardComponent implements OnInit {
  health: HealthResponse | null = null;
  loading = false;
  error: string | null = null;

  constructor(private api: ApiService) {}

  ngOnInit(): void {
    this.fetchHealth();
  }

  fetchHealth(): void {
    this.loading = true;
    this.error = null;
    this.api.getHealth().subscribe({
      next: (response) => {
        this.health = response;
        this.loading = false;
      },
      error: (err) => {
        this.error = err.message ?? 'Unable to reach the API.';
        this.loading = false;
      }
    });
  }
}
