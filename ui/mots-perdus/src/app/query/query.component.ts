import { Component } from '@angular/core';
import { CommonModule } from '@angular/common'; // Add this import
import { RouterModule } from '@angular/router'; // Import for routerLink

@Component({
  selector: 'app-query',
  imports: [CommonModule, RouterModule],
  templateUrl: './query.component.html',
  styleUrl: './query.component.css'
})
export class QueryComponent {

}
