// Copyright (C) 2020 - 2022 APC Inc.

import { Injectable } from '@angular/core';
import { IMqttMessage, MqttService } from 'ngx-mqtt';
import { Observable, Subscription } from 'rxjs';
import {IClientOptions} from 'mqtt';

export interface Foo {
    bar: string;
}

@Injectable()
export class AirplaneMqttService {

  constructor(
    private mqttService: MqttService
  ) {}

  topic(topicName: string): Observable<IMqttMessage> {
    return this.mqttService.observe(topicName);
  }

  sendmsg(topicName: string, msg: string): void {
    // use unsafe publish for non-ssl websockets
    this.mqttService.unsafePublish(topicName, msg, { qos: 1, retain: false });
  }
}
