// Copyright (C) 2020 - 2022 APC, Inc.

using System;
using System.Collections.Generic;
using CorrectionWebApp.Helper;

namespace CorrectionWebApp.Models
{
    public class SummaryModel
    {
        public List<StatusSummaryDto> StatusSummaries { get; set; }
        public int TotalImage { get; set; }
        public List<AttributeCategorySummaryDto> AttributeSummaries { get; set; }
        public List<ImageStatus> Statuses { get; set; }
        public AnnotatorTransferringSummary TransferringSummary { get; set; }
        public bool IsAdministrator { get; set; }
        public List<TrainingHistoryModel> LastTrainingStatus { get; set; }
    }

    public class AnnotatorTransferringSummary
    {
        public int TotalAlreadyConfirmedNotTransffered { get; set; }
        public int TotalStillTransferring { get; set; }
        public int TotalAlreadyTransferred { get; set; }
    }
}
