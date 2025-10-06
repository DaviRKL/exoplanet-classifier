import { NgModule } from '@angular/core';
import { RouterModule, Routes } from '@angular/router';
import { HomeComponent } from './components/home/home.component';
import { DashboardComponent } from './components/dashboard/dashboard.component';
import { ExoplanetDetailComponent } from './components/exoplanet-detail/exoplanet-detail.component';
// **ADICIONE A IMPORTAÇÃO DO NOVO COMPONENTE**
import { LearnSectionComponent } from './components/learn-section/learn-section.component';
import { PredictComponent } from './components/predict/predict.component';

const routes: Routes = [
  { path: '', component: HomeComponent, pathMatch: 'full' },
  { path: 'dashboard', component: DashboardComponent },
  { path: 'exoplanet/:name', component: ExoplanetDetailComponent },
  // **ADICIONE A NOVA ROTA AQUI**
  { path: 'learn', component: LearnSectionComponent },
  { path: 'predict', component: PredictComponent },
  { path: '**', redirectTo: '' }
];

@NgModule({
  imports: [RouterModule.forRoot(routes)],
  exports: [RouterModule]
})
export class AppRoutingModule { }