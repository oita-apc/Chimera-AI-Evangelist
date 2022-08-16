// Copyright (C) 2020 - 2022 APC, Inc.

using System;
using System.Collections.Generic;
using System.Text.Json.Serialization;
using System.Xml.Serialization;

namespace webapp.Models
{
	public class W3CWebAnnotationModel
	{
		public String Id { get; set; }
		public String Type { get; set; } = "Annotation";
		[JsonPropertyName("@context")]
		public String Context { get; set; } = "http://www.w3.org/ns/anno.jsonld";
		public List<Body> Body { get; set; }
		public Target Target { get; set; }

	}

	public class Body
    {
		public String Type { get; set; } = "TextualBody";
		public String Purpose { get; set; } = "tagging";
		public String Value { get; set; } // this is the attribute name
	}

	public class Target
    {
		public String Source { get; set; } // URL of the image
		public Selector Selector { get; set; }
	}

	public class Selector
    {
		public String Type { get; set; } // xywh --> type=FragmentSelector   points --> type=SvgSelector
		public String ConformsTo { get; set; } = "http://www.w3.org/TR/media-frags/";
		public String Value { get; set; } // this is the point data
	}


	/**
	 * <svg><polygon points="154,263 208,291 243,309 204,338 159,331 123,301 133,264"></polygon></svg>
	 */
	[XmlRoot("svg")]
	public class SvgSelector
    {
		[XmlElement("polygon")]
		public Polygon polygon { get; set; }
	}

	public class Polygon
    {
		[XmlAttribute("points")]
		public String points { get; set; }
	}
}

