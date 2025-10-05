import { Injectable } from '@angular/core';
import { of, Observable } from 'rxjs';

// Interface defining the detailed structure of an exoplanet
export interface Exoplanet {
  name: string;
  type: string;
  distance: number;
  description: string;
  icon: string;
  // Advanced Details
  planetRadius: number;
  starRadius: number;
  orbitalPeriod: number;
  transitDuration: number;
}

@Injectable({
  providedIn: 'root'
})
export class ExoplanetService {

  // Mock data updated with English descriptions
  private exoplanets: Exoplanet[] = [
    { 
      name: 'Kepler-186f', type: 'Rocky (Terrestrial)', distance: 582, 
      description: 'The first Earth-sized planet discovered in the potentially habitable zone of another star.', icon: 'public',
      planetRadius: 1.17, starRadius: 0.52, orbitalPeriod: 129.9, transitDuration: 5.03
    },
    { 
      name: 'TRAPPIST-1e', type: 'Rocky (Potentially Habitable)', distance: 40, 
      description: 'One of the most promising potentially habitable exoplanets, residing in a system of seven worlds.', icon: 'flare',
      planetRadius: 0.91, starRadius: 0.12, orbitalPeriod: 6.1, transitDuration: 0.65
    },
    { 
      name: 'Proxima Centauri b', type: 'Rocky (Super-Earth)', distance: 4.2, 
      description: 'The closest exoplanet to our solar system, but it faces extreme stellar flares.', icon: 'brightness_5',
      planetRadius: 1.07, starRadius: 0.15, orbitalPeriod: 11.2, transitDuration: 0
    },
    { 
      name: 'Kepler-452b', type: 'Rocky (Super-Earth)', distance: 1800, 
      description: 'Nicknamed "Earth\'s older cousin," it orbits a Sun-like star.', icon: 'public',
      planetRadius: 1.63, starRadius: 1.11, orbitalPeriod: 384.8, transitDuration: 10.7
    },
     { 
      name: 'HD 209458 b', type: 'Gas Giant (Hot Jupiter)', distance: 159, 
      description: 'Famously known as "Osiris," it was the first to have its atmosphere directly detected.', icon: 'cloud',
      planetRadius: 15.2, starRadius: 1.2, orbitalPeriod: 3.5, transitDuration: 3.0
    },
    { 
      name: '55 Cancri e', type: 'Rocky (Diamond Planet)', distance: 41, 
      description: 'A dense exoplanet believed to be composed largely of diamond.', icon: 'diamond',
      planetRadius: 1.87, starRadius: 0.94, orbitalPeriod: 0.7, transitDuration: 1.8
    },
    { 
      name: 'Kepler-22b', type: 'Gas Giant (Potentially Oceanic)', distance: 600, 
      description: 'A potential "water world," larger than Earth, in the habitable zone.', icon: 'water_drop',
      planetRadius: 2.4, starRadius: 0.98, orbitalPeriod: 289.9, transitDuration: 7.4
    }
  ];

  constructor() { }
  
  getAllExoplanets(): Observable<Exoplanet[]> {
    return of(this.exoplanets);
  }
  
  getExoplanetNames(): Observable<string[]> {
    return of(this.exoplanets.map(p => p.name));
  }

  getExoplanetByName(name: string): Observable<Exoplanet | undefined> {
    const cleanedName = name.trim().toLowerCase();
    const planet = this.exoplanets.find(p => p.name.trim().toLowerCase() === cleanedName);
    return of(planet);
  }
}