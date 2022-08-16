// Copyright (C) 2020 - 2022 APC Inc.

import { ComponentFixture, TestBed } from '@angular/core/testing';

import { HometabsComponent } from './hometabs.component';

describe('HometabsComponent', () => {
  let component: HometabsComponent;
  let fixture: ComponentFixture<HometabsComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      declarations: [ HometabsComponent ]
    })
    .compileComponents();
  });

  beforeEach(() => {
    fixture = TestBed.createComponent(HometabsComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
