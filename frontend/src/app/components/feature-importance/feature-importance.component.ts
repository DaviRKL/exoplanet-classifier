import { Component } from '@angular/core';
import { ApiService } from '../../services/api.service';
import { FeatureImportanceItem, FeatureImportanceResponse } from '../../models/api.models';

@Component({
  selector: 'app-feature-importance',
  templateUrl: './feature-importance.component.html',
  styleUrls: ['./feature-importance.component.css']
})
export class FeatureImportanceComponent {
  data: FeatureImportanceResponse | null = null;
  loading = false;
  error: string | null = null;
  topN = 10;
  displayedColumns = ['feature', 'importance'];

  constructor(private api: ApiService) {
    this.fetchData();
  }

  fetchData(): void {
    this.loading = true;
    this.error = null;
    this.data = null;

    this.api.getFeatureImportance(this.topN).subscribe({
      next: (response) => {
        this.data = response;
        this.loading = false;
      },
      error: (err) => {
        this.error = err.message ?? 'Unable to load feature importances.';
        this.loading = false;
      }
    });
  }

  toDataSource(items: FeatureImportanceItem[] = []): FeatureImportanceItem[] {
    return items;
  }
}
