import { Component } from '@angular/core';
import { ApiService } from '../../services/api.service';
import { TrainingResponse } from '../../models/api.models';

@Component({
  selector: 'app-train',
  templateUrl: './train.component.html',
  styleUrls: ['./train.component.css']
})
export class TrainComponent {
  selectedFile: File | null = null;
  response: TrainingResponse | null = null;
  loading = false;
  error: string | null = null;
  displayedColumns = ['model', 'accuracy', 'precision', 'recall', 'f1'];

  constructor(private api: ApiService) {}

  onFileChange(event: Event): void {
    const target = event.target as HTMLInputElement;
    if (target.files && target.files.length > 0) {
      this.selectedFile = target.files[0];
    }
  }

  onSubmit(): void {
    if (!this.selectedFile) {
      this.error = 'Select a CSV file before training.';
      return;
    }

    this.loading = true;
    this.error = null;
    this.response = null;

    this.api.train(this.selectedFile).subscribe({
      next: (res) => {
        this.response = res;
        this.loading = false;
      },
      error: (err) => {
        this.error = err.message ?? 'Training request failed.';
        this.loading = false;
      }
    });
  }

  asArray(metrics: Record<string, any>): { model: string; data: any }[] {
    return Object.entries(metrics || {}).map(([model, data]) => ({ model, data }));
  }
}
