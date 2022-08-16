// Copyright (C) 2020 - 2022 APC, Inc.

using System;
using System.ComponentModel.DataAnnotations;

namespace CorrectionWebApp.Helper
{
    public enum OrderByEnum
    {
        [Display(Name = "Image No")]
        ImageNo = 0,

        [Display(Name = "Image Date")]
        ImageDate = 1,
    }
}
