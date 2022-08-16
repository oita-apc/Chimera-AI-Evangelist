// Copyright (C) 2020 - 2022 APC, Inc.

using System;
using System.Collections.Generic;
using System.Text.Json.Serialization;

namespace webapp.Models
{
    public class ResnetV2DatasetModel 
    {
        public ResnetV2ImageList Train { get; set; }
        public ResnetV2ImageList Val { get; set; }
        public List<ResnetV2Category> Labels { get; set; }
    }

    public class ResnetV2ImageList
    {
        public List<ResnetV2Image> Images { get; set; }
    }

    public class ResnetV2Image
    {
        [JsonPropertyName("file_name")]
        public string FileName { get; set; }
        [JsonPropertyName("class_id")]
        public string ClassId { get; set; }
    }

    public class ResnetV2Category
    {
        public int Id { get; set; }
        public string Name { get; set; }
        public ResnetV2Category(int id, string name)
        {
            this.Id = id;
            this.Name = name;
        }
    }
}

