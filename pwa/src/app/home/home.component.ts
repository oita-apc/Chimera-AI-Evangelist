// Copyright (C) 2020 - 2022 APC Inc.

import { Component, OnInit } from '@angular/core';
import { Router } from '@angular/router';
import { HttpEventType, HttpResponse } from '@angular/common/http';
import { RestApiService } from './../rest-api.service';

import { MatDialog } from '@angular/material/dialog';
import { ModalComponent } from './../modal/modal.component';

@Component({
  selector: 'app-home',
  templateUrl: './home.component.html',
  styleUrls: ['./home.component.css']
})
export class HomeComponent implements OnInit {
  selectedFiles: FileList;
  progressInfos = [];

  constructor(
    private router: Router,
    private uploadService: RestApiService,
    public dialog: MatDialog
  ) { }

  ngOnInit(): void {
  }

  openDialog(): void {
    const dialogRef = this.dialog.open(ModalComponent, {
      width: '300px',
      data: {}
    });

    dialogRef.afterClosed().subscribe(result => {
      console.log('afterClosed:' + result);
      const inputNode: any = document.querySelector('#file');
      if (inputNode.files.length > 0){
        const action = result as string;
        this.progressInfos = [];
        this.selectedFiles = inputNode.files;
        if (action === 'upload') {
          this.upload(inputNode.files[0]);
        } else if (result === 'save') {
          this.saveFile(inputNode.files[0]);
        }
      }
    });
  }

  openUploader(): void {
    // this.router.navigateByUrl('/uploader');
    this.router.navigate(['uploader']);
  }

  onFileSelected(): void {
    const inputNode: any = document.querySelector('#file');
    if (inputNode.files.length > 0){
      this.progressInfos = [];
      this.selectedFiles = inputNode.files;
      // this.openDialog();
      this.upload(inputNode.files[0]);
    }
  }

  retry(): void {
    const inputNode: any = document.querySelector('#file');
    if (inputNode.files.length > 0){
      this.progressInfos = [];
      this.selectedFiles = inputNode.files;
      this.upload(inputNode.files[0]);
    }
  }

  saveFile(file: any): void {
    const reader = new FileReader();
    reader.readAsDataURL(file);
    reader.onload = () => {
      // console.log(reader.result);
      const element = document.createElement('a');
      const downloadAttr = document.createAttribute('Download');
      downloadAttr.value = 'Download';
      // element.innerHTML = 'DOWNLOAD';
      element.setAttributeNode(downloadAttr);
      element.setAttribute('href', reader.result as string);
      element.setAttribute('download', file.name);
      element.style.display = 'none';
      document.body.appendChild(element);
      element.click();
      document.body.removeChild(element);
    };
    reader.onerror = (error) => {
      console.log('Error: ', error);
    };
 }

  upload(file: any): void {
    this.progressInfos[0] = { value: 0, fileName: file.name, error: false };
    const user = this.uploadService.getCurrentLoginUser();
    this.uploadService.upload(file, user.id, 0, '').subscribe(
      event => {
        if (event.type === HttpEventType.UploadProgress) {
          this.progressInfos[0].value = Math.round(100 * event.loaded / event.total);
        } else if (event instanceof HttpResponse) {
          // this.fileInfos = this.uploadService.getFiles();
          this.progressInfos[0].value = 100;
        }
      },
      err => {
        this.progressInfos[0].error = true;
      });
  }

  logout(): void {
    this.uploadService.logout();
    this.router.navigateByUrl('/login');
  }
}
