// Copyright (C) 2020 - 2022 APC, Inc.

using System;
using System.Collections.Generic;
using CorrectionWebApp.Helper;

namespace CorrectionWebApp.Models
{
    public class DetailModel
    {
        public Guid Id { get; set; }
        public string UserName { get; set; }
        public DateTime CreatedDate { get; set; }
        public string ImageFileName { get; set; }
        public string Comment { get; set; }
        public ImageStatus Status { get; set; }
        public int ImageNo { get; set; }
        public int Width { get; set; }
        public int Height { get; set; }
        public double? Latitude { get; set; }
        public double? Longitude { get; set; }
        public List<int> Attributes { get; set; }
        public int Attribute { get; set; }
        public int? AnnotatorId { get; set; }
        public int DatasetId { get; set; }

        public List<AttributeItem> AttributeAll { get; set; } //to populate dropdown

        public Guid? PrevImageId { get; set; }
        public Guid? NextImageId { get; set; }

        public bool IsAdministrator { get; set; }
        public int ViewMode { get; set; }
        public AttributeListModel AttributeList { get;  set; }

        public string DatasetType { get; set; }
    }
}
