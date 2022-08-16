// Copyright (C) 2020 - 2022 APC Inc.

import { IItem } from './../item/interface';
import { LocalStorageService } from './../local-storage.service';
import { CItem } from './../item/controller';
import { RestApiService } from './../rest-api.service';
import { HttpEventType, HttpResponse } from '@angular/common/http';

interface FileUpload {
    file: File;
    item?: IItem;
}

export class Uploader2 {
    public uploadProgress = 0;
    public accumlatedUploadProgress = 0;
    public selectedFiles: File[];
    public successCount = 0;
    public errorCount = 0;
    public skipCount = 0;
    public totalFiles = 0; // toto file to be uploaded (after trigger upload process)
    public uploadFinishCount = 0;


    private fileUploadType = 0;
    private localStorageService: LocalStorageService;
    private uploadService: RestApiService;
    private itemController = new CItem();
    private imageSize = 'low';

    constructor(
        localStorageService: LocalStorageService,
        uploadService: RestApiService,
        fileUploadType: number
        ) {
            this.localStorageService = localStorageService;
            this.uploadService = uploadService;
            this.fileUploadType = fileUploadType;
    }

    async initiliaze(): Promise<void> {
        this.uploadProgress = 0;
        this.accumlatedUploadProgress = 0;
        this.successCount = 0;
        this.errorCount = await this.itemController.count(this.fileUploadType);
        this.skipCount = 0;
        this.totalFiles = 0;
        this.uploadFinishCount = 0;
        this.imageSize = 'low';
    }

    async calculateErrorFiles(): Promise<void> {
      this.errorCount = await this.itemController.count(this.fileUploadType);
    }

    async uploadFiles(retryErroFileOnly: boolean, selectedFiles: File[], imageSize: string): Promise<void> {
        await this.initiliaze();
        this.imageSize = imageSize;
        this.selectedFiles = selectedFiles;
        this.totalFiles = await this.itemController.count(this.fileUploadType);
        if (this.totalFiles > 0) {
          await this.itemController.updateStatusAll(this.fileUploadType, 0);
        }
        if (!retryErroFileOnly) {
            this.totalFiles += this.selectedFiles.length;
        }
        console.log('upload files totalFiles:' + this.totalFiles);
        await this.nextUpload(retryErroFileOnly);
    }

    async nextUpload(retryErroFileOnly: boolean, timeout?: number): Promise<void> {
      const nextFile = await this.getNextFile(retryErroFileOnly);
      if (nextFile) {
          if (timeout == null) {
            timeout = 500;
          }
          setTimeout(async () =>
            await this.upload(nextFile, retryErroFileOnly)
          , timeout);
      }
    }

    async getNextFile(retryErroFileOnly: boolean): Promise<FileUpload> {
        // upload
        console.log('getNextFile uploadFinishCount:' + this.uploadFinishCount);
        if (!retryErroFileOnly && this.uploadFinishCount < this.selectedFiles.length){
          const file = this.selectedFiles[this.uploadFinishCount];
          return {file, item: null};
        } else {
            const item = await this.itemController.getItem(this.fileUploadType, 0);
            if (item) {
                await this.itemController.updateStatus(item, 1);
                if (item.arrayBuffer != null) {
                  // const file: any = new Blob([item.arrayBuffer]);
                  // file.name = item.filename;
                  // file.lastModifiedDate = Number(item.filetimestamp);
                  const file = new File([item.arrayBuffer ],
                        item.filename,
                        {type: item.contentType, lastModified: Number(item.filetimestamp)});
                  console.log('getNextFile file:' + file.name + ' timestamp:' + file.lastModified);
                  return {file, item};
                }
            }
        }
        return null;
    }

    async deleteErrorFile(fileUpload: FileUpload): Promise<void> {
        if (fileUpload.item) {
            // remove from db
            console.log('remove error file id:' + fileUpload.item.id);
            await this.itemController.deleteItem(fileUpload.item);
        }
    }

    async upload(fileUpload: FileUpload, retryErroFileOnly: boolean): Promise<void> {
        const user = this.uploadService.getCurrentLoginUser();
        console.log('file:' + fileUpload.file.name + ' timestamp:' + fileUpload.file.lastModified);
        if (fileUpload.item) {
            this.errorCount -= 1;
        }
        let existingFile = false;
        if (this.fileUploadType === 1) {
          existingFile = this.localStorageService.get(fileUpload.file.lastModified + ':' + fileUpload.file.name + '');
          console.log('upload file:' + fileUpload.file.name + ' timestamp:' + fileUpload.file.lastModified);
        }
        if (existingFile){
          // already uploaded before
          this.calculateProgress(1, fileUpload.file);
          console.log('skip file:' + fileUpload.file.name + ' timestamp:' + fileUpload.file.lastModified);
          this.uploadFinishCount += 1;
          this.skipCount += 1;
          await this.deleteErrorFile(fileUpload);
          await this.nextUpload(retryErroFileOnly);
        } else {
          // TODO: resize the image depend on this.imageSize
          console.log("TODO: resize image depend on imageSize:" + this.imageSize)
          const resizedImage = await this.resizeImage(fileUpload.file, this.imageSize);
          this.uploadService.upload(resizedImage, user.id, 0, fileUpload.file.lastModified).subscribe(
            async event => {
              console.log(event);
              if (event.type === HttpEventType.UploadProgress) {
                const percentage = event.loaded / event.total;
                if (percentage < 1) {
                  this.calculateProgress(percentage, fileUpload.file);
                }
              } else if (event instanceof HttpResponse && event.status === 200) {
                var responseBody = JSON.stringify(event.body)
                var responseBodyJson = JSON.parse(responseBody);
                if(responseBodyJson["responseCode"] == 200)
                {
                  console.log('upload success file:' + fileUpload.file.name);
                  this.calculateProgress(1, fileUpload.file);
                  this.successCount += 1;
                  this.uploadFinishCount += 1;
                  this.localStorageService.set(fileUpload.file.lastModified + ':' + fileUpload.file.name + '', true);
                  await this.deleteErrorFile(fileUpload);
                  await this.nextUpload(retryErroFileOnly);
                }
                else
                {
                  console.log('upload error file:' + fileUpload.file.name + ' timestamp:' + fileUpload.file.lastModified);
                  console.log('ResresponseCodepon:' +  responseBodyJson["responseCode"] + ", ResponseMessage:" + responseBodyJson["responseMessage"]);
                  this.uploadFinishCount += 1;
                  this.errorCount += 1;
                  await this.nextUpload(retryErroFileOnly);  
                }
              }
            },
            async err => {
                this.calculateProgress(1, fileUpload.file);
                console.log('upload error file:' + fileUpload.file.name + ' timestamp:' + fileUpload.file.lastModified);
                if (fileUpload.item == null) {
                  await this.itemController.createItem(fileUpload.file, this.fileUploadType, fileUpload.file.type);
                }
                this.uploadFinishCount += 1;
                this.errorCount += 1;
                await this.nextUpload(retryErroFileOnly); // because in ios, it is frequently crashed when retrieve after create
            });
        }
      }

      calculateProgress(percentage: number, file: File): void {
        if (this.totalFiles > 0) {
          let progress0 = this.uploadFinishCount /  this.totalFiles;
          if (progress0 < 0) {
            progress0 = 0;
          }
          const progress1 = 1 / this.totalFiles * percentage;
          if (percentage === 1) {
            this.accumlatedUploadProgress = this.accumlatedUploadProgress + 1 / this.totalFiles * 100;
          } else {
            this.accumlatedUploadProgress = (progress0 + progress1) * 100;
          }
          if (this.accumlatedUploadProgress > 100) {
            this.accumlatedUploadProgress = 100;
          }
          this.uploadProgress = Math.ceil(this.accumlatedUploadProgress);
          console.log('totalFiles:' + this.totalFiles + ' progress0:' + progress0 + ' progress1:' + progress1
            + ' percentage:' + percentage + ' progress:' + this.uploadProgress + ' accumlatedProgress:'  + this.accumlatedUploadProgress
            + ' file:' + file.name
            + ' uploadFinish:' + this.uploadFinishCount);
        }
      }
    
    /* upload image in base64 with resize option */
    private resizeImage = (file:File, imageSize: string) => {
      // const file = settings.file;
      // const maxSize = settings.maxSize;
      const reader = new FileReader();
      const image = new Image();
      const canvas = document.createElement('canvas');
      const dataURItoBlob = (dataURI: string) => {
        const bytes = dataURI.split(',')[0].indexOf('base64') >= 0 ?
            atob(dataURI.split(',')[1]) :
            unescape(dataURI.split(',')[1]);
        const mime = dataURI.split(',')[0].split(':')[1].split(';')[0];
        const max = bytes.length;
        const ia = new Uint8Array(max);
        for (var i = 0; i < max; i++) ia[i] = bytes.charCodeAt(i);
        console.log("image blob size:" + max);
        return new Blob([ia], {type:mime});
      };
      const resize = () => {
        let width = image.width;
        let height = image.height;
              
        if (imageSize !== 'fullsize') {
          let maxSize = Math.max(image.height, image.width);
          if (imageSize === 'fullhd') {
            maxSize = 1920;
          } else if (imageSize === 'hd') {
            maxSize = 1280;
          } else if (imageSize === 'low') {
            maxSize = 640;
          }
          if (width > height) {
              if (width > maxSize) {
                  height *= maxSize / width;
                  width = maxSize;
              }
          } else {
              if (height > maxSize) {
                  width *= maxSize / height;
                  height = maxSize;
              }
          }
        }
        console.log('image original W:' + image.width + ' H:' + image.height);
        console.log('image reize W:' + width + ' H:' + height);
        canvas.width = width;
        canvas.height = height;
        canvas.getContext('2d').drawImage(image, 0, 0, width, height);
        let dataUrl = canvas.toDataURL('image/jpeg');
        return dataURItoBlob(dataUrl);
      };
    
      return new Promise((ok, no) => {
          if (!file.type.match(/image.*/)) {
            no(new Error("Not an image"));
            return;
          }
    
          reader.onload = (readerEvent: any) => {
            image.onload = () => ok(resize());
            image.src = readerEvent.target.result;
          };
          reader.readAsDataURL(file);
      })    
    };
}
