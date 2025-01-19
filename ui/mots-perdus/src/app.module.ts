import { NgModule } from '@angular/core';
import { BrowserModule } from '@angular/platform-browser';
import { HttpClientModule } from '@angular/common/http';
import { AppRoutingModule } from './app.routes';
import { AppComponent } from './app.component';
import { HomePageComponent } from './app/home-page/home-page.component';
import { FileUploadComponent } from './app/file-upload/file-upload.component';
import { QueryComponent } from './app/query/query.component';

@NgModule({
  imports: [
    BrowserModule,
    AppRoutingModule,
    AppComponent,
    HomePageComponent,
    FileUploadComponent,
    QueryComponent,
    HttpClientModule // Import HttpClientModule to make API calls
  ],
  providers: [],
  bootstrap: [AppComponent]
})
export class AppModule { }