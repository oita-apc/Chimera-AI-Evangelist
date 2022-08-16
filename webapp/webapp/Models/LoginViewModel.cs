// Copyright (C) 2020 - 2022 APC, Inc.

using System;
using System.ComponentModel.DataAnnotations;

namespace CorrectionWebApp.Models
{
    public class LoginViewModel
    {
        /// <summary>  
        /// Gets or sets to username address.  
        /// </summary>  
        [Required]
        [Display(Name = "ユーザーID")]
        public string Username { get; set; }

        /// <summary>  
        /// Gets or sets to password address.  
        /// </summary>  
        [Required]
        [DataType(DataType.Password)]
        [Display(Name = "パスワード")]
        public string Password { get; set; }

    }
}
