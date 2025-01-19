import { NgModule } from '@angular/core';
import { RouterModule, Routes } from '@angular/router';
import { HomePageComponent } from './app/home-page/home-page.component';
import { FileUploadComponent } from './app/file-upload/file-upload.component';
import { QueryComponent } from './app/query/query.component';

const routes: Routes = [
    { path: '', redirectTo: '/home', pathMatch: 'full' }, // Default route
    { path: 'home', component: HomePageComponent },
    { path: 'upload-file', component: FileUploadComponent },
    { path: 'query', component: QueryComponent },
    { path: '**', redirectTo: '/home' } // Wildcard route for undefined paths
  ];
@NgModule({
  imports: [RouterModule.forRoot(routes)],
  exports: [RouterModule]
})
export class AppRoutingModule {}