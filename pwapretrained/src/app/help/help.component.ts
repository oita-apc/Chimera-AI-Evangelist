// Copyright (C) 2020 - 2022 APC Inc.

declare var require: any;
import { Component, OnInit } from '@angular/core';

@Component({
  selector: 'app-help',
  templateUrl: './help.component.html',
  styleUrls: ['./help.component.css']
})
export class HelpComponent implements OnInit {
  imgname: string;
  constructor() { }

  ngOnInit(): void {
    const userAgent = navigator.userAgent;
    let ios = false;
    if (userAgent.indexOf('iPhone') >= 0) {
      ios = true;
    } else if (userAgent.indexOf('iPad') >= 0) {
      ios = true;
    }
    this.imgname = 'assets/help-android.png';
    if (ios) {
      this.imgname = 'assets/help-ios.png';
    }
    console.log('imgname:' + this.imgname);
  }

}
