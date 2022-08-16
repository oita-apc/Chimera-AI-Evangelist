CREATE TABLE `images` (
   `Id` char(36)  NOT NULL,
   `UserName` varchar(50) CHARACTER SET utf8mb4 NULL,
   `CreatedDate` datetime(6) NOT NULL,
   `ImageFileName` varchar(256) NOT NULL DEFAULT '',
   `Comment` longtext CHARACTER SET utf8mb4 NULL,
   `Status` int NOT NULL DEFAULT 0,
   `ImageNo` int NOT NULL AUTO_INCREMENT,
   `Width` int NOT NULL DEFAULT 0,
   `Height` int NOT NULL DEFAULT 0,
   `Latitude` Decimal(9,6) NULL,
   `Longitude` Decimal(9,6) NULL,
   `TransferredToAnnotation` int NOT NULL DEFAULT 0,
   `AnnotatorId` int  NULL,
    IsAnotated tinyint(1) NOT NULL,
   PRIMARY KEY (`ImageNo`)
 );