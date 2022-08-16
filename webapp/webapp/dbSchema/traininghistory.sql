CREATE TABLE `traininghistory` (
   `Id` int(10) unsigned NOT NULL AUTO_INCREMENT,
   `UserName` varchar(50) CHARACTER SET utf8mb4 NULL,
   `Status` char(15) NOT NULL,
   `ImageSize` int NOT NULL,
   `StartDate` datetime(6) NOT NULL,
   `FinishDate` datetime(6) NULL,
    `ModelType` varchar(50) NULL,
   `DatasetId` int(10) unsigned NOT NULL,
   PRIMARY KEY (`Id`)
 );