// Copyright (C) 2020 - 2022 APC, Inc.

using System;
using System.Drawing;
using System.IO;
using System.Linq;
//using ExifLib;

namespace CorrectionWebApp.Helper
{
    public static class Helper
    {        
        public static ExifData GetExifData(Stream imageStream)
        {
            return new ExifData();
            //using (var reader = new ExifReader(imageStream))
            //{
            //    Double[] latitude, longitude;
            //    var latitudeRef = "";
            //    var longitudeRef = "";

            //    var ret = new ExifData();

            //    if (reader.GetTagValue(ExifTags.GPSLatitude, out latitude)
            //         && reader.GetTagValue(ExifTags.GPSLongitude, out longitude)
            //         && reader.GetTagValue(ExifTags.GPSLatitudeRef, out latitudeRef)
            //         && reader.GetTagValue(ExifTags.GPSLongitudeRef, out longitudeRef))
            //    {
            //        var longitudeTotal = longitude[0] + longitude[1] / 60 + longitude[2] / 3600;
            //        var latitudeTotal = latitude[0] + latitude[1] / 60 + latitude[2] / 3600;
            //        ret.Latitude = (latitudeRef == "N" ? 1 : -1) * latitudeTotal;
            //        ret.Longitude = (longitudeRef == "E" ? 1 : -1) * longitudeTotal;
                    
            //    }

            //    DateTime datetimeOriginal;
                
            //    if (reader.GetTagValue(ExifTags.DateTimeOriginal, out datetimeOriginal))
            //    {
            //        ret.DateTimeOriginal = datetimeOriginal;
            //    }

            //    ushort orientation = 0;
            //    ret.orientation = orientation;
            //    if (reader.GetTagValue(ExifTags.Orientation, out orientation))
            //    {
            //        Console.WriteLine("orientation:" + orientation);
            //        ret.orientation = orientation;
            //    }

            //    return ret;                
            //}
        }

        //public static void FixImageOrientation(this Image srce, ExifData exifData)
        //{
        //    // Rotate/flip image according to <orient>
        //    switch (exifData.orientation)
        //    {
        //        case 2:
        //            srce.RotateFlip(RotateFlipType.RotateNoneFlipX);
        //            break;
        //        case 3:
        //            srce.RotateFlip(RotateFlipType.Rotate180FlipNone);
        //            break;
        //        case 4:
        //            srce.RotateFlip(RotateFlipType.Rotate180FlipX);
        //            break;
        //        case 5:
        //            srce.RotateFlip(RotateFlipType.Rotate90FlipX);
        //            break;
        //        case 6:
        //            srce.RotateFlip(RotateFlipType.Rotate90FlipNone);
        //            break;
        //        case 7:
        //            srce.RotateFlip(RotateFlipType.Rotate270FlipX);
        //            break;
        //        case 8:
        //            srce.RotateFlip(RotateFlipType.Rotate270FlipNone);
        //            break;
        //    }
        //}
    }

    

    public class ExifData
    {
        public Double? Latitude;
        public Double? Longitude;
        public DateTime? DateTimeOriginal;
        public ushort orientation;
    }
}
