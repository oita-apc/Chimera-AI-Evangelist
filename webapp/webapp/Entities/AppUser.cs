// Copyright (C) 2020 - 2022 APC, Inc.

using System;
namespace CorrectionWebApp.Entities
{
    public class AppUser
    {
        public string Id { get; set; }
        public string Name { get; set; }
        public string Password { get; set; }
        public string Role { get; set; }
    }
}
