import { Component, OnInit } from '@angular/core';
import { trigger, transition, style, animate } from '@angular/animations';

@Component({
  selector: 'app-learn-section',
  templateUrl: './learn-section.component.html',
  styleUrls: ['./learn-section.component.css'],
  animations: [
    trigger('fadeIn', [
      transition(':enter', [
        style({ opacity: 0, transform: 'translateY(20px)' }),
        animate('600ms ease-out', style({ opacity: 1, transform: 'translateY(0)' })),
      ]),
    ]),
  ]
})
export class LearnSectionComponent implements OnInit {

  constructor() { }

  ngOnInit(): void {
  }

}