// Copyright (C) 2020 - 2022 APC Inc.

import { ErrorComponent } from './error/error.component';
import { Home2Component } from './home2/home2.component';
import { HelpComponent } from './help/help.component';
import { LoginComponent } from './login/login.component';
import { NgModule } from '@angular/core';
import { Routes, RouterModule } from '@angular/router';
import { HomeComponent } from './home/home.component';
import { UploaderComponent } from './uploader/uploader.component';
import { HometabsComponent } from './hometabs/hometabs.component';
import { TestingComponent } from './testing/testing.component';
import { DocumentsComponent } from './documents/documents.component';


const routes: Routes = [
  {path: '', redirectTo: 'login', pathMatch: 'full'},
  {path: 'home', component: Home2Component},
  {path: 'login', component: LoginComponent},
  {path: 'uploader', component: UploaderComponent},
  {path: 'help', component: HelpComponent},
  {path: 'error', component: ErrorComponent},
  {path: 'hometabs', component: HometabsComponent},
  {path: 'testing', component: TestingComponent},
  {path: 'documents', component: DocumentsComponent},
  // {path: 'home2', component: Home2Component},
  // otherwise redirect to login
  { path: '**', redirectTo: 'home' }
];

@NgModule({
  imports: [RouterModule.forRoot(routes)],
  exports: [RouterModule]
})
export class AppRoutingModule { }
