// Copyright (C) 2020 - 2022 APC, Inc.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using CorrectionWebApp.Entities;
using CorrectionWebApp.Helper;

namespace CorrectionWebApp.Models
{
    public class Detail2Model
    {
        public ImageModel Image { get; set; }
        public AppImage PreviousImage { get; set; }
        public AppImage NextImage { get; set; }
    }

    public class ImageModel : AppImage
    {
        public ImageModel(AppImage image)
        {
            foreach (PropertyInfo property in typeof(AppImage).GetProperties().Where(p => p.CanWrite))
            {
                property.SetValue(this, property.GetValue(image, null), null);
            }
        }
        public int PageNo { get; set; }
    }
}
