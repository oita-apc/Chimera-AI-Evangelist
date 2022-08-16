// Copyright (C) 2020 - 2022 APC, Inc.

using System;
using System.Collections.Generic;

namespace CorrectionWebApp.Models
{
    public class LabelsModel
    {
        public List<LabelModel> Labels;
    }

    public class LabelModel
    {
        public int Id { get; set; }
        public string Name { get; set; }
        public int OrderNo { get; set; }
    }
}

