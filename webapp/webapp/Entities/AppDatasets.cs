// Copyright (C) 2020 - 2022 APC, Inc.

using System;
namespace CorrectionWebApp.Entities
{
    public class AppDatasets
    {
        public int Id { get; set; }
        public string Name { get; set; }
        public string Type { get; set; }
        public bool IsActive { get; set; }
        public bool IsTrained { get; set; }
    }
}
