// Copyright (C) 2020 - 2022 APC Inc.

import { Component, OnInit } from '@angular/core';
import { Router } from '@angular/router';
import { RestApiService } from './../rest-api.service';
import { Subscription } from 'rxjs';
import { IMqttMessage } from 'ngx-mqtt';
import { AirplaneMqttService } from './../airplane.service';
import { environment } from '../../environments/environment';
import { toBase64String } from '@angular/compiler/src/output/source_map';

@Component({
  selector: 'app-testing',
  templateUrl: './testing.component.html',
  styleUrls: ['./testing.component.css']
})
export class TestingComponent implements OnInit {

  navLinks = [
    { location: '/home', label: 'Image Collection', icon: 'add_to_photos' },
    { location: '/testing', label: 'Testing', icon: 'done' }
  ];

  sideLinks = [
    { location: '/home', label: 'Image Collection', icon: 'add_to_photos' },
    { location: '/testing', label: 'Testing', icon: 'done' },
    { location: '/documents', label: 'Documents', icon: 'book' },
  ];

  subscription: Subscription;
  clientId: number;
  isProcessing = false;
  isProcessing1 = false;
  detectionTime = '';
  detectedLabel = '';
  detectionScore  = '';
  scaleRatio = 1;
  detectedModel = '';
  detectedDataset = '';
  
  constructor(private router: Router,
    private airplaneMqtt: AirplaneMqttService,
    private uploadService: RestApiService) { 
      this.clientId = new Date().getTime();
    }

  async ngOnInit(): Promise<void> {
    this.subscribeToTopic();
    this.showActiveModel();
  }

  logout(): void {
    this.uploadService.logout();
    this.router.navigateByUrl('/login');
  }

  showActiveModel(): void{
    this.uploadService.getActiveModel().subscribe(
      (response: string) => {
        var modelstr = JSON.stringify(response)
        var modeljson = JSON.parse(modelstr);
        this.detectedDataset = modeljson["dataset"];
        this.detectedModel = this.formatActiveModel(modeljson["model"]);
      }, error => {
        console.log("get result model " + error);
        this.detectedModel = 'Server connect error';
    });      
  }

  private formatActiveModel(model:string): string{
    if (model == "detr")
      return "DETR";
    else if (model == "trident")
        return "Trident Network";
    else if (model == "faster_rcnn")
        return "Faster R-CNN";
    else if (model == "mask_rcnn")
        return "Mask R-CNN";
    else if (model == "pointrend")
        return "Mask R-CNN + PointRend";
    else if (model == "resnetv2")
        return "ResnetV2";
    else
        return "";
  }

  async onFileSelected0(): Promise<void> {
    const inputNode: any = document.querySelector('#file');
    console.log('onFileSelected count:' + inputNode.files.length);
    if (inputNode.files.length > 0){
      this.isProcessing = true;
      this.processFile(inputNode)
      const inputNode1: any = document.querySelector('#file1');
      if (inputNode1.files.length > 0){
        inputNode1.value = ''
      }
    }
  }

  async onFileSelected1(): Promise<void> {
    const inputNode: any = document.querySelector('#file1');
    console.log('onFileSelected1 count:' + inputNode.files.length);
    if (inputNode.files.length > 0){
      this.isProcessing1 = true;
      this.processFile(inputNode)
      const inputNode1: any = document.querySelector('#file');
      if (inputNode1.files.length > 0){
        inputNode1.value = ''
      }
    }
  }

  processFile(inputNode: any): void {
    console.log('file:' + inputNode.files[0].lastModified);
    const fileArray:File[] = [].slice.call(inputNode.files);
    console.log("file upload:" + fileArray[0].name);
    this.startDetection(fileArray[0]);
    this.clearResult();
  }

  startDetection(file:File): void {
    //TODO handle timeout
    
    var reader = new FileReader();
    reader.readAsDataURL(file);
    var that = this
    reader.onload = function () {
      // console.log(reader.result);
      var data = String(reader.result)
      // data = data.replace("data:image/png;base64,", "")
      data = data.substr(22);
      that.drawImage(reader.result as string) 
      // that.airplaneMqtt.sendmsg( "demoapp-v2/testingpretrained/detect/" + that.clientId + "/" + that.modeltype, data);                          
      that.airplaneMqtt.sendmsg(environment.mqtt.prefixTopic + "/testing/detect/" + that.clientId, data);      

    };
    reader.onerror = function (error) {
      console.log('Error: ', error);
    };

  }

  private drawImage(imgStr: string) {
    console.log("drawbox")
    var img = new Image();
    var canvas:any = document.getElementById("myCanvas");
    var ctx = canvas.getContext("2d");
    let that = this;
    img.onload = function ()
    {
        canvas.width = window.innerWidth;
        // calculate width and height
        var imageWidth = canvas.width;
        that.scaleRatio = 1.0 * imageWidth / img.naturalWidth;
        var imageHeight = img.naturalHeight * that.scaleRatio;
        console.log("imageWidth:" + imageWidth + " imageHeight:" + imageHeight + " scaleRatio:" + that.scaleRatio);

        // set canvas height
        canvas.height = imageHeight        

        // draw image
        ctx.drawImage(img, 0, 0, imageWidth, imageHeight);        
        console.log("img width:" + img.width + " height:" + img.height + " naturalWidth:" + img.naturalWidth + " naturalHeight:" + img.naturalHeight);        

        // that.grey(ctx, imageWidth, imageHeight);
    }
    img.src = imgStr;       
  }

  // see https://stackoverflow.com/a/37175089
  private grey(cnx, width, height) {
    var imgPixels = cnx.getImageData(0, 0, width, height);
    for(var y = 0; y < height; y++){
        for(var x = 0; x < width; x++){
            var i = (y * 4) * width + x * 4;
            var avg = (imgPixels.data[i] + imgPixels.data[i + 1] + imgPixels.data[i + 2]) / 3;
            imgPixels.data[i] = avg;
            imgPixels.data[i + 1] = avg;
            imgPixels.data[i + 2] = avg;
        }
    }

    cnx.putImageData(imgPixels, 0, 0, 0, 0, imgPixels.width, imgPixels.height);
  }

  /*
  {
    "box":[510.7365417480469,263.2699279785156,570.985595703125,295.9325866699219],
    "score":85,"label":"umbrella"
  }
  */
  private drawLabels(obj: any) {
    var canvas:any = document.getElementById("myCanvas");
    var ctx = canvas.getContext("2d");

    // draw rectangle
    ctx.strokeStyle = 'blue';
    ctx.strokeRect(
      obj.box[0] * this.scaleRatio, 
      obj.box[1] * this.scaleRatio,  
      (obj.box[2]-obj.box[0]) * this.scaleRatio, 
      (obj.box[3]-obj.box[1]) * this.scaleRatio
      );
    this.drawTextBG(
      ctx, 
      obj.label + " (" + obj.score + "%)", 
      "18px Meiryo", 
      obj.box[0] * this.scaleRatio, 
      obj.box[1] * this.scaleRatio
    );
  }

  private drawMasks(obj: any) {
    var canvas:any = document.getElementById("myCanvas");
    var ctx = canvas.getContext("2d");
    ctx.globalAlpha = 0.3;
    ctx.fillStyle = 'blue';
    ctx.beginPath();
    var toppestXY = obj.polygon[0];
    ctx.moveTo(obj.polygon[0][0] * this.scaleRatio, obj.polygon[0][1] * this.scaleRatio);
    for (var i = 1; i < obj.polygon.length -1; i++) {
      var points = obj.polygon[i];
      ctx.lineTo(points[0] * this.scaleRatio, points[1] * this.scaleRatio);
      if (toppestXY[1] > points[1])
        toppestXY = points;
    }
    
    ctx.closePath();
    ctx.fill();
    ctx.globalAlpha = 1.0;
    
    // measure text width
    var text = obj.label + " (" + obj.score + "%)";
    ctx.font = "18px Meiryo";
    var metrics = ctx.measureText(text);
    console.log('width:' + metrics.width);
    this.drawTextBG(
      ctx, 
      text, 
      ctx.font, 
      toppestXY[0] * this.scaleRatio - metrics.width/2, 
      toppestXY[1] * this.scaleRatio - 20
    );
  }

  /// expand with color, background etc.
  private drawTextBG(ctx: any, txt: string, font: string, x: number, y: number) {

    /// lets save current state as we make a lot of changes        
    ctx.save();

    // set font
    ctx.font = font;

    /// draw text from top - makes life easier at the moment
    ctx.textBaseline = 'top';

    /// color for background
    ctx.fillStyle = '#404040';
    
    /// get width of text
    var width = ctx.measureText(txt).width;

    /// draw background rect assuming height of font
    ctx.fillRect(x, y, width, parseInt(font, 10));
    
    /// text color
    ctx.fillStyle = '#fff';

    /// draw text on top
    ctx.fillText(txt, x, y);
    
    /// restore original state
    ctx.restore();
  }

  private formatDate(date: Date): string {
    const month = '' + (date.getMonth() + 1);
    const day = '' + date.getDate();
    const year = '' + date.getFullYear();

    var hours = '' + date.getHours();
    if(Number(hours) < 10)
    {
      hours = "0" + hours
    }
    var minutes = '' + date.getMinutes();
    if(Number(minutes) < 10)
    {
      minutes = "0" + minutes
    }
    var seconds = '' + date.getSeconds();
    if(Number(seconds) < 10)
    {
      seconds = "0" + seconds
    }

    return year + '年' + month + '月' + day + '日 ' + hours + ':' + minutes + ':' + seconds;
  }

  private subscribeToTopic(): void {
    // this.subscription = this.airplaneMqtt.topic('demoapp-v2/testingpretrained/result/' + this.clientId)
    this.subscription = this.airplaneMqtt.topic(environment.mqtt.prefixTopic + '/testing/result/' + this.clientId)
        .subscribe((data: IMqttMessage) => {
          
          this.isProcessing = false;
          this.isProcessing1 = false;
          
          // console.log(data.payload.toString());
          this.detectionTime = this.formatDate(new Date());
          const result = JSON.parse(data.payload.toString());
          // console.log('mqtt item:' + JSON.stringify(result, null, 2));
          if (result.numberOfObjects == 0) {
            this.detectedLabel = "(can not detect any object)";
          } else {          
            const _modelType = result.modelType;
            if (_modelType == 'faster_rcnn' ||
                _modelType == 'trident' ||
                _modelType == 'detr') {
                  this.showObjectDetectionResult(result.objects);
            } else if (_modelType == 'mask_rcnn' ||
                _modelType == 'pointrend') {
                var canvas:any = document.getElementById("myCanvas");
                var ctx = canvas.getContext("2d");
                this.grey(ctx, canvas.width, canvas.height);
                this.showImageSegmentationResult(result.objects);
            } else if (_modelType == 'resnetv2') {
                this.showImageClassificationResult(result);
            } 
          }
          this.showActiveModel();
        });
  }

  private async showObjectDetectionResult(objs: any) {
    for (var i = 0; i < objs.length; i++) {
      var obj = objs[i];
      console.log("OD obj", JSON.stringify(obj));

      await this.uploadService.getLabel(obj.label).toPromise()
      .then((response: string) => {
        var labelstr = JSON.stringify(response)
        var labeljson = JSON.parse(labelstr);
        var label = labeljson["label"];
        obj.label = label;
        this.drawLabels(obj);
      })
        .catch((error: any) => {
          console.log("get result label " + error);
      });
    }
  }

  private async showImageSegmentationResult(objs: any) {
    for (var i = 0; i < objs.length; i++) {
      var obj = objs[i];
      console.log("IS obj", JSON.stringify(obj));

      await this.uploadService.getLabel(obj.label).toPromise()
      .then((response: string) => {
        var labelstr = JSON.stringify(response)
        var labeljson = JSON.parse(labelstr);
        var label = labeljson["label"];
        obj.label = label;
        this.drawMasks(obj);
      })
        .catch((error: any) => {
          console.log("get result label " + error);
      });
    }
  }

  private showImageClassificationResult(result: any) {
    if(result.label == "0") {
      this.detectedLabel = "UNKNOWN";
    } else if(result.label == "-1")  {
      this.detectedLabel = "UNTRAINED";
    } else {
      try {
        this.uploadService.getLabel(result.label).subscribe(
          (response: string) => {
            var labelstr = JSON.stringify(response)
            var labeljson = JSON.parse(labelstr);
            var label = labeljson["label"];

            this.detectedLabel = label;
          }, error => {
            console.log("get result label " + error);
            this.detectedLabel = "UNKNOWN";
          }
        );
      } catch (err) {
        this.detectedLabel = "UNKNOWN";
      }
    }
    this.detectionScore = '(' + result.score + '%)';
  }

  private clearResult():void {
    this.detectionTime = '';
    this.detectedLabel = '';
    this.detectionScore = '';
  }

}
