import { NgModule } from '@angular/core';
import { RouterModule, Routes } from '@angular/router';
import { DashboardComponent } from './components/dashboard/dashboard.component';
import { FeatureImportanceComponent } from './components/feature-importance/feature-importance.component';
import { MetricsComponent } from './components/metrics/metrics.component';
import { PredictComponent } from './components/predict/predict.component';
import { TrainComponent } from './components/train/train.component';
import { HomeComponent } from './components/home/home.component';

const routes: Routes = [
  { path: '', component: HomeComponent },
  { path: 'dashboard', component: DashboardComponent },
  { path: 'train', component: TrainComponent },
  { path: 'metrics', component: MetricsComponent },
  { path: 'feature-importance', component: FeatureImportanceComponent },
  { path: '**', redirectTo: '' }
];

@NgModule({
  imports: [RouterModule.forRoot(routes)],
  exports: [RouterModule]
})
export class AppRoutingModule {}
