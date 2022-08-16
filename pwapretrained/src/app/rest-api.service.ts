// Copyright (C) 2020 - 2022 APC Inc.

import { User } from './_model/User';
import { Injectable } from '@angular/core';
import { HttpClient, HttpHeaders, HttpErrorResponse, HttpRequest, HttpEvent } from '@angular/common/http';
import { environment } from '../environments/environment';
import { Observable, of } from 'rxjs';
import { catchError, map, tap } from 'rxjs/operators';


@Injectable({
  providedIn: 'root'
})
export class RestApiService {
  private baseUrl: string;
  constructor(private http: HttpClient) {
    this.baseUrl = environment.apiBaseURL;
  }

  login (user: User): Observable<User> {
    const httpOptions = {
      headers: new HttpHeaders({ 'Content-Type': 'application/json' })
    };
    // console.log('user -> ' + JSON.stringify(user, null, 4));
    return this.http.post<User>(this.baseUrl + '/api/ImageUploader/Login',
      user, httpOptions).pipe(
        tap((x: User) => console.log(`login w/ id=${x.id}`)),
        catchError(this.handleError<User>('login'))
    );
  }

  upload(file: File, userName, attribute, imagedate): Observable<HttpEvent<any>> {
    const formData: FormData = new FormData();

    formData.append('file', file);

    const req = new HttpRequest('POST',
        `${this.baseUrl}/api/ImageUploader/UploadImage?userName=${userName}&attributes=${attribute}&timestamp=${imagedate}`,
        formData, {
      reportProgress: true,
      responseType: 'json'
    });

    return this.http.request(req);
  }

  logout(): void {
    console.log('remove "currentUser" from localStorage');
    // remove user from local storage to log user out
    localStorage.removeItem('currentUser');
  }

  getCurrentLoginUser(): User {
    const user = JSON.parse(localStorage.getItem('currentUser'));
    // console.log('current user -> ' + JSON.stringify(user, null, 4));
    return user;
  }

  /**
   * Handle Http operation that failed.
   * Let the app continue.
   * @param operation - name of the operation that failed
   * @param result - optional value to return as the observable result
   */
  private handleError<T> (operation = 'operation', result?: T): any {
    return (error: any): Observable<T> => {

      // TODO: send the error to remote logging infrastructure
      console.error(error); // log to console instead

      // TODO: better job of transforming error for user consumption
      console.log(`${operation} failed: ${error.message}`);

      // Let the app keep running by returning an empty result.
      return of(result as T);
    };
  }
}
