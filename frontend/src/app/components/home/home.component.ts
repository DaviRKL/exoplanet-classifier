import { Component, AfterViewInit, ElementRef, QueryList, ViewChildren } from '@angular/core';
import { trigger, state, style, animate, transition } from '@angular/animations';

@Component({
  selector: 'app-home',
  templateUrl: './home.component.html',
  styleUrls: ['./home.component.css'],
  animations: [
    trigger('scrollAnimation', [
      state('out', style({
        opacity: 0,
        transform: 'translateY(50px)' // Começa um pouco mais para baixo
      })),
      state('in', style({
        opacity: 1,
        transform: 'translateY(0)' // Termina na posição original
      })),
      transition('out => in', [
        animate('0.8s ease-out') // Duração e tipo de animação
      ]),
    ])
  ]
})
export class HomeComponent implements AfterViewInit {
  // Pega todos os elementos marcados com #scrollTarget
  @ViewChildren('scrollTarget') scrollTargets!: QueryList<ElementRef>;

  // Armazena o estado de animação ('in' ou 'out') para cada elemento
  public states: { [key: number]: 'in' | 'out' } = {};
  private observer?: IntersectionObserver;

  ngAfterViewInit(): void {
    // Inicializa o estado de todos os alvos de animação como 'out'
    this.scrollTargets.forEach((_, index) => {
      this.states[index] = 'out';
    });
    this.initIntersectionObserver();
  }

  private initIntersectionObserver(): void {
    const options = {
      threshold: 0.2 // A animação dispara quando 20% do elemento está visível
    };

    this.observer = new IntersectionObserver((entries) => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          const targetIndex = this.scrollTargets.toArray().findIndex(
            (el) => el.nativeElement === entry.target
          );
          if (targetIndex !== -1) {
            // Muda o estado para 'in', o que aciona a animação do Angular
            this.states[targetIndex] = 'in';
            // Para a observação após a primeira vez para não repetir a animação
            this.observer?.unobserve(entry.target);
          }
        }
      });
    }, options);

    // Começa a observar cada elemento alvo
    this.scrollTargets.forEach(target => {
      this.observer?.observe(target.nativeElement);
    });
  }
}