import { Component } from '@angular/core';
import { CommonModule } from '@angular/common'; // Add this import
import { RouterModule } from '@angular/router'; // Import for routerLink
import { FormsModule } from '@angular/forms'; 

@Component({
  selector: 'app-file-upload',
  imports: [CommonModule, RouterModule, FormsModule],
  templateUrl: './file-upload.component.html',
  styleUrl: './file-upload.component.css'
})
export class FileUploadComponent {

  public selectedOption: string = '';
  public authorName: string = '';
  public bookTitle : string = '';

}
