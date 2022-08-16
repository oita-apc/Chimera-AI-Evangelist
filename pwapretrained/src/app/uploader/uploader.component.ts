// Copyright (C) 2020 - 2022 APC Inc.

import { LocalStorageService } from './../local-storage.service';
import { RestApiService } from './../rest-api.service';
import { Component, OnInit } from '@angular/core';
import { HttpEventType, HttpResponse } from '@angular/common/http';
import { Router } from '@angular/router';

@Component({
  selector: 'app-uploader',
  templateUrl: './uploader.component.html',
  styleUrls: ['./uploader.component.css']
})
export class UploaderComponent implements OnInit {
  selectedFiles: FileList;
  progressInfos = [];
  btnUploadEnabled = false;
  uploadFinishCount = 0;
  maxFiles = 100;
  errorMsg = '';

  get uploadErrorCount(): number {
    const errorCounts = this.progressInfos.filter(x => x.error);
    return errorCounts.length;
  }

  constructor(
    private router: Router,
    private localStorageService: LocalStorageService,
    private uploadService: RestApiService) { }

  ngOnInit(): void {
    this.btnUploadEnabled = false;
    this.uploadFinishCount = 0;
  }

  upload(idx: number, retry: boolean): void {
    const user = this.uploadService.getCurrentLoginUser();
    if (idx >= this.selectedFiles.length){
        return; // already completed
    }
    const file = this.selectedFiles[idx];
    console.log('file:' + file.name + ' timestamp:' + file.lastModified + ' retry:' + retry);

    if (!retry){
      // this is initialization
      this.progressInfos[idx] = { value: 0, fileName: file.name, error: false, uploadedBefore: false };
    } else {
      if (!this.progressInfos[idx].error || this.progressInfos[idx].uploadedBefore) {
        this.uploadFinishCount += 1;
        this.upload(idx + 1, retry);
        return;
      }
    }

    const existingFile = this.localStorageService.get(file.name);
    if (existingFile){
      // already uploaded before
      this.uploadFinishCount += 1;
      this.progressInfos[idx].error = false;
      this.progressInfos[idx].uploadedBefore = true;
      this.upload(idx + 1, retry);
    } else {
      this.uploadService.upload(file, user.id, 0, file.lastModified).subscribe(
        event => {
          if (event.type === HttpEventType.UploadProgress) {
            const progress = Math.round(100 * event.loaded / event.total);
            if (progress <= 99) {
              this.progressInfos[idx].value = Math.round(100 * event.loaded / event.total);
            }
          } else if (event instanceof HttpResponse) {
            // this.fileInfos = this.uploadService.getFiles();
            this.progressInfos[idx].value = 100;
            this.progressInfos[idx].error = false;
            this.uploadFinishCount += 1;
            this.localStorageService.set(file.name, true);
            this.upload(idx + 1, retry);
          }
        },
        err => {
          this.uploadFinishCount += 1;
          this.progressInfos[idx].error = true;
          this.upload(idx + 1, retry);
        });
    }
  }

  uploadFiles(): void {
    if (this.selectedFiles.length > 0){
      this.btnUploadEnabled = false;
      this.upload(0, false);
    }
  }

  retry(): void {
    this.uploadFinishCount = 0;
    if (this.selectedFiles.length > 0){
      this.btnUploadEnabled = false;
      this.upload(0, true);
    }
  }


  onFileSelected(): void {
    const inputNode: any = document.querySelector('#file');
    if (inputNode.files.length > this.maxFiles){
      this.errorMsg = 'Maximum number of files is ' + this.maxFiles;
    } else
    if (inputNode.files.length > 0){
      this.errorMsg = '';
      this.uploadFinishCount = 0;
      this.progressInfos = [];
      this.selectedFiles = inputNode.files;
      this.btnUploadEnabled = true;
    }
  }

  gotoHome(): void {
    this.router.navigate(['home']);
  }

  logout(): void {
    this.uploadService.logout();
    this.router.navigateByUrl('/login');
  }
}
