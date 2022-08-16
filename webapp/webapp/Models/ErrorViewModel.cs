// Copyright (C) 2020 - 2022 APC, Inc.

using System;

namespace CorrectionWebApp.Models
{
    public class ErrorViewModel
    {
        public string RequestId { get; set; }

        public bool ShowRequestId => !string.IsNullOrEmpty(RequestId);
    }
}
