// Copyright (C) 2020 - 2022 APC, Inc.

using System;
using CorrectionWebApp.Helper;

namespace CorrectionWebApp.Models
{
    public class StatusSummaryDto
    {
        public ImageStatus Status { get; set; }
        public int Count { get; set; }
    }
}
