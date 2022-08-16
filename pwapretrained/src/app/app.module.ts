
// Copyright (C) 2020 - 2022 APC Inc.

import { BrowserModule } from '@angular/platform-browser';
import { NgModule } from '@angular/core';

import { MatToolbarModule} from '@angular/material/toolbar';
import { MatIconModule } from '@angular/material/icon';
import { MatCardModule} from '@angular/material/card';
import { MatButtonModule} from '@angular/material/button';
import { MatProgressBarModule} from '@angular/material/progress-bar';
import { MatProgressSpinnerModule} from '@angular/material/progress-spinner';
import { MatInputModule } from '@angular/material/input';
import { MatDialogModule } from '@angular/material/dialog';
import { MatTabsModule } from '@angular/material/tabs';

import { HttpClientModule } from '@angular/common/http';

import { FormsModule, ReactiveFormsModule } from '@angular/forms';

import { MatSidenavModule } from '@angular/material/sidenav';
import { MatListModule } from '@angular/material/list';

import { AppRoutingModule } from './app-routing.module';
import { AppComponent } from './app.component';
import { ServiceWorkerModule } from '@angular/service-worker';
import { environment } from '../environments/environment';
import { BrowserAnimationsModule } from '@angular/platform-browser/animations';
import { HomeComponent } from './home/home.component';
import { UploaderComponent } from './uploader/uploader.component';

import { NgxMatFileInputModule } from '@angular-material-components/file-input';
import { LoginComponent } from './login/login.component';
import { ModalComponent } from './modal/modal.component';
import { HelpComponent } from './help/help.component';
import { Home2Component } from './home2/home2.component';
import { ErrorComponent } from './error/error.component';
import { FooterComponent } from './footer/footer.component';
import { HometabsComponent } from './hometabs/hometabs.component';
import { TestingComponent } from './testing/testing.component';
import { DocumentsComponent } from './documents/documents.component';
import { AirplaneMqttService } from './airplane.service';

import {IMqttServiceOptions, MqttModule} from 'ngx-mqtt';

import { OverlayModule } from '@angular/cdk/overlay';

const MQTT_SERVICE_OPTIONS: IMqttServiceOptions = {
  hostname: environment.mqtt.server,
  port: environment.mqtt.port,
  protocol: (environment.mqtt.protocol === 'wss') ? 'wss' : 'ws',
  path: '/mqtt'
};

@NgModule({
  declarations: [
    AppComponent,
    HomeComponent,
    UploaderComponent,
    LoginComponent,
    ModalComponent,
    HelpComponent,
    Home2Component,
    ErrorComponent,
    FooterComponent,
    HometabsComponent,
    TestingComponent,
    DocumentsComponent    
  ],
  imports: [
    FormsModule,
    ReactiveFormsModule,
    BrowserModule,
    AppRoutingModule,
    ServiceWorkerModule.register('ngsw-worker.js', { enabled: environment.production }),
    BrowserAnimationsModule,
    MatToolbarModule,
    MatTabsModule,
    MatIconModule,
    MatCardModule,
    MatButtonModule,
    MatProgressBarModule,
    MatProgressSpinnerModule,
    OverlayModule,
    MatInputModule,
    MatSidenavModule,
    MatListModule,
    HttpClientModule,
    NgxMatFileInputModule,
    MatDialogModule,
    MqttModule.forRoot(MQTT_SERVICE_OPTIONS)
  ],
  providers: [AirplaneMqttService],
  bootstrap: [AppComponent]
})
export class AppModule { }
