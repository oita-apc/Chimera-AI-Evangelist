// Copyright (C) 2020 - 2022 APC, Inc.

using System;
using System.Collections.Generic;

namespace CorrectionWebApp.Models
{
    public class AnnotatorHelperModel
    {
        public List<ImageInfo> Images { get; set; }
        public int TotalRecords { get; set; }
    }

    public class ImageInfo
    {
        public Guid Id { get; set; }
        public string FileName { get; set; }
        public int ImageNo { get; set; }
    }
}
