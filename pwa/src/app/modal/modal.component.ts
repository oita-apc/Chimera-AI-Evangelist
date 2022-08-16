// Copyright (C) 2020 - 2022 APC Inc.

import { Component, OnInit, Inject } from '@angular/core';
import { MatDialogRef } from '@angular/material/dialog';

@Component({
  selector: 'app-modal',
  templateUrl: './modal.component.html',
  styleUrls: ['./modal.component.css']
})
export class ModalComponent implements OnInit {
  constructor(
    public dialogRef: MatDialogRef<ModalComponent>
  ) {}

  ngOnInit(): void {
  }

  onSaveClick(): void {
    this.dialogRef.close('save');
  }

  onUploadClick(): void {
    this.dialogRef.close('upload');
  }
}
