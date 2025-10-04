export interface HealthResponse {
  message: string;
  model_loaded: boolean;
}

export interface PredictionPayload {
  koi_period: number;
  koi_duration: number;
  koi_depth: number;
  koi_prad: number;
  koi_srad: number;
  koi_smass: number;
  koi_model_snr: number;
  koi_impact: number;
}

export interface PredictionResponse {
  prediction: string;
}

export interface TrainingMetrics {
  accuracy: number;
  precision: number;
  recall: number;
  f1: number;
  confusion_matrix: number[][];
  classes: string[];
}

export interface TrainingResponse {
  best_model: string;
  metrics: Record<string, TrainingMetrics>;
}

export interface FeatureImportanceItem {
  feature: string;
  importance: number;
}

export interface FeatureImportanceResponse {
  model: string;
  importances: FeatureImportanceItem[];
}

export interface PlotResponse {
  confusion_matrix: string;
  feature_importance: string;
}
