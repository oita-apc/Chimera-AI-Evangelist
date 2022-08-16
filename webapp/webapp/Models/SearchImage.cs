// Copyright (C) 2020 - 2022 APC, Inc.

using System;
using System.Collections.Generic;

namespace CorrectionWebApp.Models
{
    public class SearchImage
    {
        public List<string> SearchAttributes { get; set; }
        public string SearchAttribute { get; set; }
        public int? SearchStatus { get; set; }
        public string SearchPhotographer { get; set; }
        public string DateStartStr { get; set; }
        public string DateEndStr { get; set; }
        public string SearchComment { get; set; }
        public int PageNo { get; set; } = 1;
        public int ViewMode { get; set; } = 1;
        public bool? Submitted { get; set; }
        public int PageSize { get; set; }

        public int? ImageNoFrom { get; set; }
        public int? ImageNoEnd { get; set; }
        public bool AlreadyTransferred { get; set; } = false;
        public int OrderByField { get; set; } = 0;

        public bool? IsAnnotated { get; set; }

        public AttributeListModel AttributeList { get; set; }
    }
}
