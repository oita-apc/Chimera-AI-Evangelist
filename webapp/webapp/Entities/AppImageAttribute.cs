// Copyright (C) 2020 - 2022 APC, Inc.

using System;
namespace CorrectionWebApp.Entities
{
    public class AppImageAttribute
    {
        public int Id { get; set; }
        public Guid ImageId { get; set; }
        public int AttributeId { get; set; }
        public Guid? AnnotationId { get; set; }
        public string AnnotationType { get; set; }
        public string Data { get; set; }
        public int DatasetId { get; set; }
    }
}
