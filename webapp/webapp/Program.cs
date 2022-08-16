// Copyright (C) 2020 - 2022 APC, Inc.

using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Hosting;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Routing;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Net.Http;
using System.Net.Http.Json;
using System.Threading;
using System.Threading.Tasks;
using CorrectionWebApp.Services;
using Microsoft.AspNetCore.Authentication.Cookies;
using Microsoft.AspNetCore.HttpOverrides;
using Swashbuckle.AspNetCore.Filters;
using CorrectionWebApp.swagger;
using Microsoft.OpenApi.Models;

var  MyAllowSpecificOrigins = "_myAllowSpecificOrigins";
var builder = WebApplication.CreateBuilder(args);


//builder.Configuration.AddJsonFile("appsettings.json");
//builder.Host.ConfigureAppConfiguration((hostingContext, config) =>
//{
//    config.AddJsonFile("appsettings.json", optional: false, reloadOnChange: true);
//});

// Add services to the container.
//builder.Services.AddRazorPages();
builder.Services.AddAuthentication(options =>
{
    options.DefaultSignInScheme = CookieAuthenticationDefaults.AuthenticationScheme;
    options.DefaultAuthenticateScheme = CookieAuthenticationDefaults.AuthenticationScheme;
    options.DefaultChallengeScheme = CookieAuthenticationDefaults.AuthenticationScheme;
}).AddCookie(options =>
{
    options.LoginPath = new PathString("/Home/Index");
    options.ExpireTimeSpan = TimeSpan.FromHours(8.0);
});

builder.Services.AddCors(options =>
{
    options.AddPolicy(name: MyAllowSpecificOrigins,
                      builder =>
                      {
                          builder.WithOrigins("http://localhost:4200",
                          "http://192.168.11.27:4200")
                          .AllowAnyHeader()
                          .AllowAnyMethod();
                      });
});

builder.Services.AddAppServices(builder.Configuration);

builder.Services.AddControllersWithViews();
builder.Services.Configure<ForwardedHeadersOptions>(options =>
{
    options.ForwardedHeaders =
        ForwardedHeaders.XForwardedFor | ForwardedHeaders.XForwardedProto;
});

// Register the Swagger generator, defining 1 or more Swagger documents
builder.Services.AddSwaggerGen(c =>
{
    // specify operation filter for upload file
    c.OperationFilter<AddFileOperationFilter>();

    c.SwaggerDoc("v1", new OpenApiInfo { Title = "My API", Version = "v1" });
    c.OperationFilter<AddResponseHeadersFilter>(); // [SwaggerResponseHeader]
    c.OperationFilter<AppendAuthorizeToSummaryOperationFilter>(); // Adds "(Auth)" to the summary so that you can see which endpoints have Authorization
                                                                  // or use the generic method, e.g. c.OperationFilter<AppendAuthorizeToSummaryOperationFilter<MyCustomAttribute>>();

    // add Security information to each operation for OAuth2
    c.OperationFilter<SecurityRequirementsOperationFilter>();
    // or use the generic method, e.g. c.OperationFilter<SecurityRequirementsOperationFilter<MyCustomAttribute>>();

    // if you're using the SecurityRequirementsOperationFilter, you also need to tell Swashbuckle you're using OAuth2
    c.AddSecurityDefinition("oauth2", new OpenApiSecurityScheme
    {
        Description = "Standard Authorization header using the Bearer scheme. Example: \"bearer {token}\"",
        In = ParameterLocation.Header,
        Name = "Authorization",
        Type = SecuritySchemeType.ApiKey
    });
});

builder.Services.AddDistributedMemoryCache(); // Adds a default in-memory implementation of IDistributedCache
builder.Services.AddSession();


var app = builder.Build();

// Configure the HTTP request pipeline.
if (app.Environment.IsDevelopment())
{
    app.UseDeveloperExceptionPage();
}
else
{
    //app.UseExceptionHandler("/Error");
    app.UseExceptionHandler("/Home/Error");
    // The default HSTS value is 30 days. You may want to change this for production scenarios, see https://aka.ms/aspnetcore-hsts.
    app.UseHsts();
}

app.UseStaticFiles();

// Enable middleware to serve generated Swagger as a JSON endpoint.
app.UseSwagger();

// Enable middleware to serve swagger-ui (HTML, JS, CSS, etc.),
// specifying the Swagger JSON endpoint.
app.UseSwaggerUI(c =>
{
    c.SwaggerEndpoint("/swagger/v1/swagger.json", "My API V1");
});

app.UseRouting();

// Make sure you call this before calling app.UseMvc()
app.UseCors(MyAllowSpecificOrigins);

app.UseSession();



//app.UseAuthorization();
app.UseAuthentication();

//app.MapRazorPages();

app.UseEndpoints(endpoints =>
{
    endpoints.MapControllerRoute(
        name: "default",
        pattern: "{controller=Home}/{action=Index}/{id?}");
});

app.Run();




