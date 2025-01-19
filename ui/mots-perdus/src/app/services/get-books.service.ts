import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { AddedBook } from '../models/added-book';
import { ApiResponse } from '../models/api-response';
import { Observable } from 'rxjs';
import { AppConfig } from '../../app.config';

@Injectable({
  providedIn: 'root'
})
export class GetBooksService {

  private apiUrl        = AppConfig.apiUrl;
  private booksEndPoint = this.apiUrl + '/books';

  constructor(private http: HttpClient) { }

  getAllBooks(): Observable<ApiResponse<AddedBook[]>> {
    return this.http.get<ApiResponse<AddedBook[]>>(this.booksEndPoint)
  }
}
