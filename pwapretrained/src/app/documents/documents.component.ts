// Copyright (C) 2020 - 2022 APC Inc.

import { Component, OnInit } from '@angular/core';
import { Router } from '@angular/router';
import { RestApiService } from '../rest-api.service';

@Component({
  selector: 'app-testing',
  templateUrl: './documents.component.html',
  styleUrls: ['./documents.component.css']
})
export class DocumentsComponent implements OnInit {

  sideLinks = [
//    { location: '/home', label: 'Image Collection', icon: 'add_to_photos' },
    { location: '/testing', label: 'Testing', icon: 'done' },
    { location: '/documents', label: 'Documents', icon: 'book' },
  ];

  clientId: number;

  constructor(private router: Router,
    private uploadService: RestApiService) { 
      this.clientId = new Date().getTime();
  }

  ngOnInit(): void {
  }

  logout(): void {
    this.uploadService.logout();
    this.router.navigateByUrl('/login');
  }
}
