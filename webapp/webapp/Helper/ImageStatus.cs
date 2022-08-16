// Copyright (C) 2020 - 2022 APC, Inc.

using System;
using System.ComponentModel.DataAnnotations;

namespace CorrectionWebApp.Helper
{
    public enum ImageStatus
    {
        [Display(Name = "NOT-ENTERED")]
        NotEntered = 0,

        [Display(Name = "UN-CONFIRMED")]
        Unconfirmed = 1,

        [Display(Name = "CONFIRMED")]
        Confirmed = 2,

        [Display(Name = "REJECTED")]
        Rejected = 3
    }
    
}
