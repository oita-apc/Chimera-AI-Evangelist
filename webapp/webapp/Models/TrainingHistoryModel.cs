// Copyright (C) 2020 - 2022 APC, Inc.

using System;
namespace CorrectionWebApp.Models
{
    public class TrainingHistoryModel
    {
        public string UserName { get; set; }
        public string Status { get; set; }
        public String ImageSize { get; set; }
        public DateTime? StartDate { get; set; }
        public DateTime? FinishDate { get; set; }
        public string ModelType { get; set; }
        public string DatasetName { get; set; }
        public bool IsTrained { get; set; }
        public bool IsActive { get; set; }
        public int DatasetId { get; set; }

        public string getModelType()
        {
            if (ModelType == "detr")
                return "DETR";
            else if (ModelType == "trident")
                return "Trident Network";
            else if (ModelType == "faster_rcnn")
                return "Faster R-CNN";
            else if (ModelType == "mask_rcnn")
                return "Mask R-CNN";
            else if (ModelType == "pointrend")
                return "Mask R-CNN + PointRend";
            else if (ModelType == "resnetv2")
                return "ResnetV2";
            else
                return "";
        }
    }
}
