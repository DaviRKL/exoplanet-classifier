import { Injectable } from '@angular/core';
import { HttpClient, HttpErrorResponse, HttpParams } from '@angular/common/http';
import { Observable, catchError, throwError } from 'rxjs';
import { environment } from '../../environments/environment';
import {
  FeatureImportanceResponse,
  HealthResponse,
  PlotResponse,
  PredictionPayload,
  PredictionResponse,
  TrainingResponse
} from '../models/api.models';

@Injectable({ providedIn: 'root' })
export class ApiService {
  private readonly baseUrl = environment.apiBaseUrl;

  constructor(private http: HttpClient) {}

  getHealth(): Observable<HealthResponse> {
    return this.http
      .get<HealthResponse>(`${this.baseUrl}/`)
      .pipe(catchError((error) => this.handleError(error)));
  }

  predict(payload: PredictionPayload): Observable<PredictionResponse> {
    return this.http
      .post<PredictionResponse>(`${this.baseUrl}/predict`, payload)
      .pipe(catchError((error) => this.handleError(error)));
  }

  train(file: File): Observable<TrainingResponse> {
    const formData = new FormData();
    formData.append('file', file, file.name);

    return this.http
      .post<TrainingResponse>(`${this.baseUrl}/train`, formData)
      .pipe(catchError((error) => this.handleError(error)));
  }

  getMetrics(): Observable<TrainingResponse> {
    return this.http
      .get<TrainingResponse>(`${this.baseUrl}/metrics`)
      .pipe(catchError((error) => this.handleError(error)));
  }

  getFeatureImportance(topN: number = 10): Observable<FeatureImportanceResponse> {
    const params = new HttpParams().set('top_n', topN);
    return this.http
      .get<FeatureImportanceResponse>(`${this.baseUrl}/feature-importance`, { params })
      .pipe(catchError((error) => this.handleError(error)));
  }

  getPlots(topN: number = 10): Observable<PlotResponse> {
    const params = new HttpParams().set('top_n', topN);
    return this.http
      .get<PlotResponse>(`${this.baseUrl}/plots`, { params })
      .pipe(catchError((error) => this.handleError(error)));
  }

  private handleError(error: HttpErrorResponse) {
    console.error('API error', error);
    return throwError(() => new Error(error.message || 'API request failed'));
  }
}

