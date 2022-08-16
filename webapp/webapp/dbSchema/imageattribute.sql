CREATE TABLE `imageattribute` (
   `Id` int(10) unsigned NOT NULL AUTO_INCREMENT,
   `ImageId` char(36)  NOT NULL,
   `AttributeId` int(10) unsigned NOT NULL ,
    `AnnotationId` char(36)  NULL,
    `AnnotationType` varchar(50)   NULL,
    `Data` mediumtext   NULL,
    `DatasetId` int(10) unsigned NOT NULL,
   PRIMARY KEY (`Id`)
 );