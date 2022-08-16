// Copyright (C) 2020 - 2022 APC Inc.

import { User } from './../_model/User';
import { Component, OnInit } from '@angular/core';
import { FormBuilder, FormGroup, Validators } from '@angular/forms';
import { ActivatedRoute, Router } from '@angular/router';
import { RestApiService } from './../rest-api.service';
import { environment } from 'src/environments/environment';

@Component({
  selector: 'app-login',
  templateUrl: './login.component.html',
  styleUrls: ['./login.component.css']
})
export class LoginComponent implements OnInit {
  form: FormGroup;
  public loginInvalid: boolean;
  public loading = false;
  private formSubmitAttempt: boolean;
  private returnUrl: string;

  constructor(
    private fb: FormBuilder,
    private route: ActivatedRoute,
    private router: Router,
    private uploadService: RestApiService,
  ) { }

  ngOnInit(): void {
    this.returnUrl = this.route.snapshot.queryParams.returnUrl || '/home';

    this.form = this.fb.group({
      username: ['', Validators.required],
      password: ['', Validators.required]
    });
  }

  async onSubmit(): Promise<void> {
    this.loading = true;
    this.loginInvalid = false;
    this.formSubmitAttempt = false;
    if (this.form.valid) {
      try {
        const username = this.form.get('username').value;
        const password = this.form.get('password').value;
        console.log('try to login username:' + username + ' password:' + password);
        const model = new User();
        model.id = username;
        model.password = password;
        this.uploadService.login(model)
            .subscribe(
              user => {
                if (typeof user === 'object' && typeof user.id !== 'undefined' && user.id != null) {
                  var now = new Date();
                  var expires = environment.expires;
                  now.setHours(now.getHours() + expires);
                  user.expires = now.toString();
                  localStorage.setItem('currentUser', JSON.stringify(user));
                  // store user details in local storage to keep user logged in between pages
                  localStorage.setItem('currentUser', JSON.stringify(user));
                  // console.log('login result user -> ' + JSON.stringify(user, null, 4));
                  this.router.navigateByUrl('/testing');
                } else {
                  console.log('login result user unknown -> ' + JSON.stringify(user, null, 4));
                  this.loading = false;
                  this.loginInvalid = true;
                }
              },
              error => {
                console.log('login result user -> ' + error);
                this.loginInvalid = true;
                this.loading = false;
              });
      } catch (err) {
        this.loginInvalid = true;
      }
    } else {
      this.formSubmitAttempt = true;
    }
  }
}
