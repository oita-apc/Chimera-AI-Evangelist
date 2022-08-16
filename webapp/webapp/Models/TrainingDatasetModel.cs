// Copyright (C) 2020 - 2022 APC, Inc.

using System;
using System.Collections.Generic;

namespace CorrectionWebApp.Models
{
    public class TrainingDatasetModel
    {
        public List<int> TrainAnnotatorIds { get; set; }
        public List<int> ValAnnotatorIds { get; set; }
    }
}
