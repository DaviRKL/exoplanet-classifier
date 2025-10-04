import { Component } from '@angular/core';
import { FormBuilder, FormGroup, Validators } from '@angular/forms';
import { ApiService } from '../../services/api.service';
import { PredictionResponse } from '../../models/api.models';

@Component({
  selector: 'app-predict',
  templateUrl: './predict.component.html',
  styleUrls: ['./predict.component.css']
})
export class PredictComponent {
  readonly fields = [
    { name: 'koi_period', label: 'koi_period' },
    { name: 'koi_duration', label: 'koi_duration' },
    { name: 'koi_depth', label: 'koi_depth' },
    { name: 'koi_prad', label: 'koi_prad' },
    { name: 'koi_srad', label: 'koi_srad' },
    { name: 'koi_smass', label: 'koi_smass' },
    { name: 'koi_model_snr', label: 'koi_model_snr' },
    { name: 'koi_impact', label: 'koi_impact' }
  ];

  form: FormGroup;
  loading = false;
  prediction: PredictionResponse | null = null;
  error: string | null = null;

  constructor(private fb: FormBuilder, private api: ApiService) {
    this.form = this.fb.group({
      koi_period: [365.25, [Validators.required, Validators.min(0)]],
      koi_duration: [12.4, [Validators.required, Validators.min(0)]],
      koi_depth: [0.0018, [Validators.required, Validators.min(0)]],
      koi_prad: [1.4, [Validators.required, Validators.min(0)]],
      koi_srad: [0.95, [Validators.required, Validators.min(0)]],
      koi_smass: [0.9, [Validators.required, Validators.min(0)]],
      koi_model_snr: [45, [Validators.required, Validators.min(0)]],
      koi_impact: [0.45, [Validators.required, Validators.min(0)]]
    });
  }

  onSubmit(): void {
    if (this.form.invalid) {
      this.form.markAllAsTouched();
      return;
    }

    this.loading = true;
    this.prediction = null;
    this.error = null;

    this.api.predict(this.form.value).subscribe({
      next: (response) => {
        this.prediction = response;
        this.loading = false;
      },
      error: (err) => {
        this.error = err.message ?? 'Failed to retrieve the prediction.';
        this.loading = false;
      }
    });
  }
}
