CREATE TABLE `datasets` (
   `Id` int  NOT NULL,
   `Name` varchar(128) CHARACTER SET utf8mb4 NOT NULL,
    `Type` varchar(8) NOT NULL COMMENT 'ic:image classification od: object detection is:image segmentation',
    `IsActive` tinyint(1) NOT NULL  DEFAULT 0,
    `IsTrained` tinyint(1) NOT NULL  DEFAULT 0,
   PRIMARY KEY (`Id`)
 );
