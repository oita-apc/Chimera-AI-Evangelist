// Copyright (C) 2020 - 2022 APC, Inc.

using System;
namespace CorrectionWebApp.swagger
{
    /// <summary>
    /// don't forget to set Parameter property to field name
    /// </summary>
    [AttributeUsage(AttributeTargets.Method, Inherited = false, AllowMultiple = false)]
    public class SwaggerUploadFile : Attribute
    {
        public string Parameter { get; set; }
        public string Description { get; set; } = "Select a file to upload";
        public string Example { get; set; }
    }
}
