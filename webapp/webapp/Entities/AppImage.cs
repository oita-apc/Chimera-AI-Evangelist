// Copyright (C) 2020 - 2022 APC, Inc.

using System;
using System.ComponentModel.DataAnnotations;
using System.ComponentModel.DataAnnotations.Schema;

namespace CorrectionWebApp.Entities
{
    public class AppImage
    {
        [Key]
        [DatabaseGenerated(DatabaseGeneratedOption.Identity)]
        public int ImageNo { get; set; }

        public Guid Id { get; set; }
        public string UserName { get; set; }
        public DateTime CreatedDate { get; set; }
        public string ImageFileName { get; set; }
        public string Comment { get; set; }
        public int Status { get; set; }        
        public int Width { get; set; }
        public int Height { get; set; }
        public double? Latitude { get; set; }
        public double? Longitude { get; set; }
        public int TransferredToAnnotation { get; set; }
        public int? AnnotatorId { get; set; }
        public bool IsAnotated { get; set; }
    }
}
