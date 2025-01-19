import { Component, OnInit } from '@angular/core';
import { AddedBook } from '../models/added-book';
import { GetBooksService } from '../services/get-books.service';
import { CommonModule } from '@angular/common'; // Add this import
import { RouterModule } from '@angular/router'; // Import for routerLink

@Component({
  selector: 'app-home-page',
  imports: [CommonModule, RouterModule],
  templateUrl: './home-page.component.html',
  styleUrl: './home-page.component.css'
})
export class HomePageComponent implements OnInit {

  // get list of all existing books in db 
  public allBooks: AddedBook[] = [];
  public message: string ='';

  constructor(private getBookService: GetBooksService) {}

  ngOnInit(): void {
    this.fetchBooks();
  }

  fetchBooks(): void{
    this.getBookService.getAllBooks().subscribe({
      
      next: (response) => {
        this.allBooks = response.data;

        this.message = response.message
        if (response.errors) {
          console.error('Error fetching all books;', response.errors)
        }      
      },
      error: (error) => {

        console.error('Network error:', error)
        this.message = 'Failed to connect to the server. Please try again.'
      }
  
  });
} 

}
