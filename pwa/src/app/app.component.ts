// Copyright (C) 2020 - 2022 APC Inc.

import { RestApiService } from './rest-api.service';
import { User } from './_model/User';
import { Component, OnInit } from '@angular/core';
import { Router, NavigationEnd } from '@angular/router';
import { SwUpdate } from '@angular/service-worker';
import {Platform} from '@angular/cdk/platform';
import { environment } from 'src/environments/environment';


@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.css']
})
export class AppComponent implements OnInit {
  title = 'correction-app';
  isDataLoaded = false;
  user: User;
  currentUrl = '/';

  constructor(
    private adminService: RestApiService,
    private router: Router,
    private swUpdate: SwUpdate,
    private platform: Platform,
    private uploadService: RestApiService
  ) {

    // tslint:disable-next-line:no-string-literal
    const isInWebAppiOS = ('standalone' in window.navigator) && (window.navigator['standalone']);
    const isInWebAppChrome = (window.matchMedia('(display-mode: standalone)').matches);
    console.log('isInWebAppiOS:' + isInWebAppiOS
      + ' isInWebAppChrome:' + isInWebAppChrome + ' navigator.onLine:' + navigator.onLine);
    if (isInWebAppiOS || isInWebAppChrome) {
      console.log('display-mode is standalone');
    } else {
      // if (navigator.onLine === true) {
      //   if ((platform.ANDROID && platform.BLINK) ||
      //       (platform.IOS && platform.SAFARI && !this.isChromeOnIos())
      //   ) {
      //     this.router.navigateByUrl('/help');
      //   } else {
      //     this.router.navigateByUrl('/error');
      //   }

      //   console.log('display-mode is NOT standalone');
      //   return;
      // }
    }

    // check login state
    this.router.events.subscribe(
      (e) => {
        // this is main logic to control login state
        if (e instanceof NavigationEnd) {
          this.currentUrl = e.url;
          this.user = this.adminService.getCurrentLoginUser();
          console.log('user:' + this.user);
          console.log('url:' + e.url);

          if(this.user != null)
          {
            if(this.user.expires == null){
              this.uploadService.logout();
              this.router.navigateByUrl('/login');
            }
            if(new Date(this.user.expires) > new Date())
            {
              var now = new Date();
              var expires = environment.expires;
              now.setHours(now.getHours() + expires);
              this.user.expires = now.toString();
              console.log('update expires:' + this.user.expires)
              localStorage.setItem('currentUser', JSON.stringify(this.user));
            }
            else
            {
              // timeout
              this.uploadService.logout();
              this.router.navigateByUrl('/login');
            }
          }

          if (this.currentUrl === '/login') {
            if (this.user != null) {
              // redirect to home
              this.router.navigateByUrl('/home');
            }
          }  else if (this.currentUrl === '/') {
            if (this.user != null) {
              // redirect to home
              this.router.navigateByUrl('/home');
            }
          }
          else {
            this.isDataLoaded = true;
            if (this.currentUrl !== '/login' && this.user == null) {
              // redirect to login
              console.log('redirect to login');
              this.router.navigateByUrl('/login');
              return;
            }
          }
        }
      }
    );
  }

  isChromeOnIos(): boolean {
    if(/CriOS/i.test(navigator.userAgent) &&
    /iphone|ipod|ipad/i.test(navigator.userAgent)){
      return true;
    } else {
      return false;
    }
  }

  ngOnInit(): void {
    console.log('swUpdate:' + this.swUpdate.isEnabled);
    if (this.swUpdate.isEnabled) {
      this.swUpdate.available.subscribe(() => {
          console.log('swUpdate try to refresh ');
          window.location.reload();
      });
    }
  }
}
