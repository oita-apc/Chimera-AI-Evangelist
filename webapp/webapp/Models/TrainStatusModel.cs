// Copyright (C) 2020 - 2022 APC, Inc.

using System;
using System.Collections.Generic;

namespace CorrectionWebApp.Models
{
    public class TrainStatusModel
    {
        public string status { get; set; }
        public GpuInfo gpuInfo { get; set; }
        public string finish_date { get; set; }
        public string model_type { get; set; }
        public string dataset_id { get; set; }
    }

    public class GpuInfo
    {
        public string Total { get; set; }
        public string Used { get; set; }
        public string Free { get; set; }
    }
}
