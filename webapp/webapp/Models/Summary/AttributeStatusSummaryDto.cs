// Copyright (C) 2020 - 2022 APC, Inc.

using System;
using System.Collections.Generic;
using CorrectionWebApp.Helper;

namespace CorrectionWebApp.Models
{
    public class AttributeCategorySummaryDto
    {
        public string Name { get; set; }
        public int Id { get; set; }
        public int OrderNo { get; set; }
        public List<AttributeStatusSummaryDto> Attributes { get; set; }
    }
    public class AttributeStatusSummaryDto
    {
        public int AttributeId { get; set; }
        public string AttributeName { get; set; }
        public List<StatusSummaryDto> StatusSummaries { get; set; }
        public int TotalImage { get; set; }
    }
}
