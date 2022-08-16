// Copyright (C) 2020 - 2022 APC Inc.

import { Component, OnInit } from '@angular/core';
import { Router } from '@angular/router';
import { RestApiService } from './../rest-api.service';
import { MatToolbarModule } from '@angular/material/toolbar';

@Component({
  selector: 'app-hometabs',
  templateUrl: './hometabs.component.html',
  styleUrls: ['./hometabs.component.css']
})
export class HometabsComponent implements OnInit {
  // activeLink = './home'
  // active = true
  navLinks = [
    { location: '/home', label: 'Image Collection', icon: 'camera_alt' },
    { location: '/shared', label: 'Overview', icon: 'account_circle' }
  ];
  
  constructor(
    private router: Router,
    private uploadService: RestApiService
  ) {}

  ngOnInit(): void {
  }

  logout(): void {
    this.uploadService.logout();
    this.router.navigateByUrl('/login');
  }
}
