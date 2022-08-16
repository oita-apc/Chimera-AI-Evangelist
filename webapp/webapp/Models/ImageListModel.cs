// Copyright (C) 2020 - 2022 APC, Inc.

using System;
using System.Collections.Generic;
using CorrectionWebApp.Helper;

namespace CorrectionWebApp.Models
{
    public class ImageListModel
    {
        public List<ImageItem> Images { get; set; }
        public int TotalCount { get; set; }
        public SearchImage SearchImage { get; set; }
        public AttributeListModel AttributeList { get; set; }
        public bool IsAdministrator { get; internal set; }

        public static List<int> ThumbnailPageSize = new List<int> { 18, 24, 36, 48, 60};
        public static List<int> ListPageSize = new List<int> { 10, 20, 30, 40, 50 };
    }

    public class PageInfo
    {
        public int PageNo { get; set; }
        public int TotalPageNo { get; set; }
        public int TotalCount { get; set; }
    }

    public class ImageItem
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
        public int? AnnotatorId { get; set; }
        public List<AttributeItem> Attributes { get; set; }
    }

    public class AttributeItem
    {
        public int Id { get; set; }
        public string Name { get; set; }
    }
}
