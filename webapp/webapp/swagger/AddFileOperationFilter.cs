// Copyright (C) 2020 - 2022 APC, Inc.

using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.OpenApi.Models;
using Swashbuckle.AspNetCore.SwaggerGen;

namespace CorrectionWebApp.swagger
{
    /// <summary>
    ///  see https://dejanstojanovic.net/aspnet/2019/august/handling-file-uploads-in-swagger-ui-with-aspnet-core/
    ///  especially on the comment section by "Jason Law"
    /// </summary>
    public class AddFileOperationFilter : IOperationFilter
    {
        public void Apply(OpenApiOperation operation, OperationFilterContext context)
        {
            if (!(operation?.RequestBody?.Content?.Any(x => x.Key.ToLower() == "multipart/form-data") ?? false))
                return;

            var uploadFiles = context.MethodInfo.DeclaringType.GetCustomAttributes(true)
                .Union(context.MethodInfo.GetCustomAttributes(true)).OfType<SwaggerUploadFile>();
            if (uploadFiles.Count() == 0) return;

            var uploadFile = uploadFiles.First();
            operation.RequestBody.Content["multipart/form-data"].Schema.Properties =
                new Dictionary<string, OpenApiSchema>
                {
                    [uploadFile.Parameter] = new OpenApiSchema
                    {
                        Type = "string",
                        Format = "binary",
                        Description = uploadFile.Description
                    }
                };

        }
    }
}
