// Copyright (C) 2020 - 2022 APC, Inc.

using System;
using System.Collections.Generic;
using System.Text.Json.Serialization;

namespace CorrectionWebApp.Models
{
    public class CocoAllDatasetModel
    {
        public CocoDatasetModel Train { get; set; }
        public CocoDatasetModel Val { get; set; }
        public List<CategoryCoco> Labels { get; set; }
    }
	public class CocoDatasetModel
	{
        public List<ImageCoco> Images { get; set; }
        public List<AnnotationCoco> Annotations { get; set; }
        public List<CategoryCoco> Categories { get; set; }
	}

    public class ImageCoco
    {
        //public int License { get; set; }
        [JsonPropertyName("file_name")]
        public string FileName { get; set; }
        public int Height { get; set; }
        public int Width { get; set; }
        public int Id { get; set; }
    }
    public class AnnotationCoco
    {
        public int Id { get; set; }
        [JsonPropertyName("image_id")]
        public int ImageId { get; set; }
        [JsonPropertyName("category_id")]
        public int CategoryId { get; set; }
        public int Area { get; set; }
        public Boolean Iscrowd { get; set; } = false;
        public Boolean Isbbox { get; set; } = false;
        public List<List<int>> Segmentation { get; set; }
        public List<int> Bbox { get; set; }
        //public AnnotationCoco(int id, int imageId, int categoryId, int area)
        //{
        //    this.Id = id;
        //    this.ImageId = imageId;
        //    this.CategoryId = categoryId;
        //    this.Area = area;
        //}
    }

    public class CategoryCoco
    {
        public int Id { get; set; }
        public string Name { get; set; }
        public string Supercategory { get; set; } = "";
        public CategoryCoco(int id, string name)
        {
            this.Id = id;
            this.Name = name;
            this.Supercategory = "";
        }
    }
}

