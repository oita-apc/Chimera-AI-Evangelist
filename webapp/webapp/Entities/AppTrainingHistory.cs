// Copyright (C) 2020 - 2022 APC, Inc.

using System;
namespace CorrectionWebApp.Entities
{
    public class AppTrainingHistory
    {
        public int Id { get; set; }
        public string UserName { get; set; }
        public string Status { get; set; }
        public int ImageSize { get; set; }
        public DateTime StartDate { get; set; }
        public DateTime? FinishDate { get; set; }
        public string ModelType { get; set; }
        public int DatasetId { get; set; }
    }
}
