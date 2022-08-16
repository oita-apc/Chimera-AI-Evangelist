// Copyright (C) 2020 - 2022 APC, Inc.

using System;
namespace CorrectionWebApp.Helper
{
    public static class Const
    {
        public const int PAGE_SIZE_THUMBNAIL = 24;
        public const int PAGE_SIZE_LIST = 22;
        public const string TRAINING_STATUS_RUNNING = "RUNNING";
        public const string TRAINING_STATUS_COMPLETED = "COMPLETED";
        public const string TRAINING_STATUS_CANCELLED = "CANCELLED";

        public const string OBJECT_DETECTION = "od";
        public const string INSTANCE_SEGMENTATION = "is";
        public const string IMAGE_CLASSIFICATION = "ic";
    }
}
