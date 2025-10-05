import { Component, OnInit } from '@angular/core';
import { ActivatedRoute, Router } from '@angular/router';
import { Exoplanet, ExoplanetService } from '../../services/exoplanet.service';
import { trigger, transition, style, animate, state } from '@angular/animations';

@Component({
  selector: 'app-exoplanet-detail',
  templateUrl: './exoplanet-detail.component.html',
  styleUrls: ['./exoplanet-detail.component.css'],
  animations: [
    trigger('pageTransition', [
      transition(':enter', [
        style({ opacity: 0 }),
        animate('500ms ease-out', style({ opacity: 1 })),
      ]),
    ]),
    trigger('scrollAnimation', [
      state('out', style({ opacity: 0, transform: 'translateY(40px)' })),
      state('in', style({ opacity: 1, transform: 'translateY(0)' })),
      transition('out => in', animate('0.8s 0.2s ease-out'))
    ]),
    // Animação para o modo avançado
    trigger('advancedMode', [
      transition(':enter', [
        style({ opacity: 0, height: 0 }),
        animate('400ms ease-out', style({ opacity: 1, height: '*' })),
      ]),
      transition(':leave', [
        animate('400ms ease-in', style({ opacity: 0, height: 0 }))
      ])
    ])
  ]
})
export class ExoplanetDetailComponent implements OnInit {
  planet: Exoplanet | undefined;
  loading = true;
  error = false;
  animationState = 'out';
  isAdvancedMode = false;

  constructor(
    private route: ActivatedRoute,
    private router: Router,
    private exoplanetService: ExoplanetService
  ) { }

  ngOnInit(): void {
    const planetName = this.route.snapshot.paramMap.get('name');
    if (planetName) {
      this.exoplanetService.getExoplanetByName(planetName).subscribe(data => {
        if (data) {
          this.planet = data;
          setTimeout(() => this.animationState = 'in', 100);
        } else {
          this.error = true;
        }
        this.loading = false;
      });
    } else {
      this.error = true;
      this.loading = false;
    }
  }

  goBack(): void {
    this.router.navigate(['/dashboard']);
  }
}