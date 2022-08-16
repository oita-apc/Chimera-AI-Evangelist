// Copyright (C) 2020 - 2022 APC, Inc.

using System;
using System.Collections.Generic;

namespace CorrectionWebApp.Models
{
    public class AttributeListModel
    {
        public const int MAX_LENGTH_TITLE = 40;
        public List<AttributeCategory> AttributeList { get; set; }
        public string getSelectedAttributesTitle()
        {
            var titles = new List<string>();
            foreach(var cat in AttributeList)
            {
                foreach (var attr in cat.Attributes)
                {
                    if (attr.Checked)
                        titles.Add(attr.Name);
                }
            }

            var titlestr = string.Join(", ", titles);

            if (titles.Count == 0)
            {
                return "ALL";
            } else if (titlestr.Length > MAX_LENGTH_TITLE)
            {
                return titlestr.Substring(0, MAX_LENGTH_TITLE) + " ...";
            } else
            {
                return titlestr;
            }
        }
    }

    public class AttributeCategory
    {
        public int Id { get; set; }
        public string Name { get; set; }
        public int OrderNo { get; set; }
        public bool Checked { get; set; }
        public List<Attribute> Attributes { get; set; }
    }

    public class Attribute
    {
        public int Id { get; set; }
        public string Name { get; set; }
        public int OrderNo { get; set; }
        public bool Checked { get; set; }
    }
}
