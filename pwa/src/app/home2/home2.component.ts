// Copyright (C) 2020 - 2022 APC Inc.

import { Uploader2 } from './uploader2';
import { CItem } from './../item/controller';
import { LocalStorageService } from './../local-storage.service';
import { Component, OnInit } from '@angular/core';
import { Router } from '@angular/router';
import { RestApiService } from './../rest-api.service';
import { HttpEventType, HttpResponse } from '@angular/common/http';

@Component({
  selector: 'app-home2',
  templateUrl: './home2.component.html',
  styleUrls: ['./home2.component.css']
})
export class Home2Component implements OnInit {
  btnUploadEnabled1 = false;
  btnUploadEnabled0 = false;
  uploader0: Uploader2;
  uploader1: Uploader2;
  selectedFiles1: FileList;
  imageSize = 'low'

  navLinks = [
    { location: '/home', label: 'Image Collection', icon: 'add_to_photos' },
    { location: '/testing', label: 'Testing', icon: 'done' }
  ];

  sideLinks = [
    { location: '/home', label: 'Image Collection', icon: 'add_to_photos' },
    { location: '/testing', label: 'Testing', icon: 'done' },
    { location: '/documents', label: 'Documents', icon: 'book' },
  ];

  constructor(
    private router: Router,
    private localStorageService: LocalStorageService,
    private uploadService: RestApiService
  ) {
    this.uploader0 = new Uploader2(this.localStorageService, this.uploadService, 0);
    this.uploader1 = new Uploader2(this.localStorageService, this.uploadService, 1);
  }

  async onItemChange(value: string): Promise<void>{
    this.imageSize = value;
    console.log("imageSize:" + this.imageSize);
  }

  disabledAllButtons(): void {
    this.btnUploadEnabled0 = false;
    this.btnUploadEnabled1 = false;
  }

  async onFileSelected0(): Promise<void> {
    const inputNode: any = document.querySelector('#file');
    console.log('onFileSelected count:' + inputNode.files.length + ' imageSize:' + this.imageSize);
    if (inputNode.files.length > 0){
      console.log('file:' + inputNode.files[0].lastModified);
      const fileArray = [].slice.call(inputNode.files);
      await this.uploader0.uploadFiles(false, fileArray, this.imageSize);
    }
  }

  async retry0(): Promise<void> {
    if (this.uploader0.errorCount > 0){
      const inputNode: any = document.querySelector('#file');
      await this.uploader0.uploadFiles(true, inputNode.files, this.imageSize);
    }
  }

  async onFileSelected1(): Promise<void> {
    await this.uploader1.initiliaze();
    const inputNode: any = document.querySelector('#files');
    this.selectedFiles1 = inputNode.files;
    await this.uploader1.calculateErrorFiles();
    console.log('onFileSelected1 count:' + this.selectedFiles1.length + ' errorFiles:' +  this.uploader1.totalFiles + ' imageSize:' + this.imageSize);
    if (this.selectedFiles1 && this.selectedFiles1.length > 0) {
      this.btnUploadEnabled1 = true;
    }
  }

  async retry1(): Promise<void> {
    if (this.uploader1.errorCount > 0){
      this.disabledAllButtons();
      const inputNode: any = document.querySelector('#file');
      // this #file will be empty but we need it to simulate empty array
      await this.uploader1.uploadFiles(true, inputNode.files, this.imageSize);
    }
  }

  async uploadFiles1(): Promise<void> {
    if (this.selectedFiles1 && this.selectedFiles1.length > 0) {
      this.disabledAllButtons();
      const fileArray = [].slice.call(this.selectedFiles1);
      const sortedFileArray = fileArray.sort((a, b)  => {
        return a.lastModified - b.lastModified;
      });
      for (let index = 0; index < sortedFileArray.length; index++) {
        console.log('item index:' + index + ' item:' + sortedFileArray[index].lastModified);
      }
      await this.uploader1.uploadFiles(false, sortedFileArray, this.imageSize);
    }
  }

  async ngOnInit(): Promise<void> {
    await this.uploader0.initiliaze();
    await this.uploader1.initiliaze();
  }

  logout(): void {
    this.uploadService.logout();
    this.router.navigateByUrl('/login');
  }
}
