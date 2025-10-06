import { Component } from '@angular/core';
import { FormBuilder, FormGroup, Validators } from '@angular/forms';
import { ApiService } from '../../services/api.service';
import { HttpClient } from '@angular/common/http';

@Component({
  selector: 'app-predict',
  templateUrl: './predict.component.html',
  styleUrls: ['./predict.component.css'],
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
  fileName = '';
  parsedRows: Array<Record<string, string>> = [];
  headers: string[] = [];
  previewLimit = 5;
  loading = false;
  // allow null to avoid assigning null to a string-typed field
  error: string | null = null;

  // keep CSV-batch predictions as an array (may be used by CSV flow)
  predictions: Array<{ row: Record<string, string>; prediction: string; probability: number }> = [];

  // add a single prediction object used by the current template
  prediction: { prediction: string; probability?: number; summary?: string } | null = null;

  constructor(private fb: FormBuilder, private api: ApiService, private http: HttpClient) {
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
    // reset template-facing prediction and previous batch predictions
    this.prediction = null;
    this.predictions = [];
    this.error = null;

    this.api.predict(this.form.value).subscribe({
      next: (response) => {
        const resAny: any = response;
        // normalize a few possible response shapes:
        // - direct object { prediction: "...", probability: ... }
        // - { predictions: [ { prediction, probability }, ... ] }
        // - array [ { prediction, probability }, ... ]
        if (Array.isArray(resAny) && resAny.length > 0) {
          this.prediction = resAny[0];
        } else if (resAny && Array.isArray(resAny.predictions) && resAny.predictions.length > 0) {
          this.prediction = resAny.predictions[0];
        } else {
          // assume it's a direct prediction object (or null)
          this.prediction = resAny || null;
        }
        this.loading = false;
      },
      error: (err) => {
        this.error = err?.message ?? 'Failed to retrieve the prediction.';
        this.loading = false;
      }
    });
  }

  onFileSelected(event: Event) {
    this.error = '';
    const input = event.target as HTMLInputElement;
    if (!input.files || !input.files.length) return;
    const file = input.files[0];
    if (!file.name.toLowerCase().endsWith('.csv')) {
      this.error = 'Please upload a .csv file';
      return;
    }
    this.fileName = file.name;
    this.parseCsvFile(file);
  }

  private parseCsvFile(file: File) {
    const reader = new FileReader();
    reader.onload = () => {
      const text = String(reader.result || '');
      try {
        const { headers, rows } = this.parseCSV(text);
        this.headers = headers;
        this.parsedRows = rows;
        // clear previous predictions
        this.predictions = [];
      } catch (err) {
        this.error = 'Error parsing CSV';
      }
    };
    reader.onerror = () => {
      this.error = 'Error reading file';
    };
    reader.readAsText(file);
  }

  // Basic CSV parser that handles quoted fields. Returns headers and array of row objects.
  private parseCSV(text: string): { headers: string[]; rows: Array<Record<string, string>> } {
    const lines = text.split(/\r\n|\n/).filter(l => l.trim().length > 0);
    if (lines.length === 0) return { headers: [], rows: [] };

    const parseLine = (line: string) => {
      const result: string[] = [];
      let cur = '';
      let inQuotes = false;
      for (let i = 0; i < line.length; i++) {
        const ch = line[i];
        if (ch === '"' ) {
          if (inQuotes && line[i + 1] === '"') { cur += '"'; i++; continue; }
          inQuotes = !inQuotes;
          continue;
        }
        if (ch === ',' && !inQuotes) {
          result.push(cur);
          cur = '';
          continue;
        }
        cur += ch;
      }
      result.push(cur);
      return result;
    };

    const headers = parseLine(lines[0]).map(h => h.trim());
    const rows = lines.slice(1).map(line => {
      const cols = parseLine(line);
      const obj: Record<string, string> = {};
      headers.forEach((h, idx) => {
        obj[h] = cols[idx] !== undefined ? cols[idx].trim() : '';
      });
      return obj;
    });

    return { headers, rows };
  }

  // send original CSV to backend; expects response { predictions: [{ prediction, probability }] }
  sendCsv() {
    if (!this.fileName || !this.parsedRows.length) {
      this.error = 'No CSV to send';
      return;
    }
    this.loading = true;
    this.error = '';
    // build FormData with original CSV content recreated from parsedRows
    const csv = this.buildCsvFromParsed();
    const blob = new Blob([csv], { type: 'text/csv' });
    const form = new FormData();
    form.append('file', blob, this.fileName);

    // adjust endpoint if different in your backend
    this.http.post<{ predictions: Array<{ prediction: string; probability: number }> }>('/api/predict/csv', form)
      .subscribe({
        next: (res) => {
          this.predictions = (res.predictions || []).map((p, i) => ({
            row: this.parsedRows[i] || {},
            prediction: p.prediction,
            probability: p.probability ?? 0,
          }));
          this.loading = false;
        },
        error: (err) => {
          this.loading = false;
          this.error = 'Prediction request failed';
        },
      });
  }

  private buildCsvFromParsed(): string {
    const cols = this.headers;
    const rows = this.parsedRows.map(r => cols.map(c => {
      const v = (r[c] ?? '').toString();
      if (v.includes(',') || v.includes('"') || v.includes('\n')) {
        return `"${v.replace(/"/g, '""')}"`;
      }
      return v;
    }).join(','));
    return [cols.join(','), ...rows].join('\n');
  }
}
