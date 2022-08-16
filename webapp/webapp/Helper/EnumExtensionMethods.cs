// Copyright (C) 2020 - 2022 APC, Inc.

using System;
using System.Reflection;
using System.Linq;
namespace CorrectionWebApp.Helper
{
    public static class EnumExtensionMethods
    {
        public static string GetDescription(this Enum GenericEnum)
        {
            Type genericEnumType = GenericEnum.GetType();
            MemberInfo[] memberInfo = genericEnumType.GetMember(GenericEnum.ToString());
            if ((memberInfo != null && memberInfo.Length > 0))
            {
                var _Attribs = memberInfo[0].GetCustomAttributes(typeof(System.ComponentModel.DataAnnotations.DisplayAttribute), false);
                if ((_Attribs != null && _Attribs.Count() > 0))
                {
                    return ((System.ComponentModel.DataAnnotations.DisplayAttribute)_Attribs.ElementAt(0)).Name;
                }
            }
            return GenericEnum.ToString();
        }

        public static string ToStringJpn(this DateTime datetime)
        {
            return String.Format("{0:0000}年{1:00}月{2:00}日 {3:00}:{4:00}",
                datetime.Year, datetime.Month, datetime.Day, datetime.Hour, datetime.Minute);
        }

        public static string ToStringJpn(this DateTime? datetime)
        {
            if(datetime == null) {
                return "";
            }
            DateTime result = (DateTime) datetime;
            return result.ToStringJpn();
        }
    }
}
