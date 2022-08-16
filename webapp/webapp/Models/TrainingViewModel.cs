// Copyright (C) 2020 - 2022 APC, Inc.

using System;
using System.Collections.Generic;

namespace CorrectionWebApp
{
    public class TrainingViewModel
    {
        public List<String> Labels { get; set; }
        public int NumberOfData { get; set; }
        public string DatasetType { get; set; }
    }
}

