// Copyright (C) 2020 - 2022 APC, Inc.

using System;
using System.Collections.Generic;
using CorrectionWebApp.Helper;

namespace CorrectionWebApp.Models
{
    public class DatasetsModel
    {
        public List<Dataset> Datasets { get; set; }
    }
    public class Dataset
    {
        public int Id { get; set; }
        public string Name { get; set; }
        public string Type { get; set; }
        public bool IsActive { get; set; }
        public bool IsTrained { get; set; }

        public string getModelTypeLongName()
        {
            if (this.Type == Const.IMAGE_CLASSIFICATION) {
                return "Image Classification";
            } else if (this.Type == Const.OBJECT_DETECTION)
            {
                return "Object Detection";
            } else if (this.Type == Const.INSTANCE_SEGMENTATION)
            {
                return "Image Segmentation";
            }
            return "";
        }
    }
}
