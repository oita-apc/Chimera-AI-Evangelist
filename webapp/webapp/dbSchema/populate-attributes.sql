delete from attributecategories;
delete from attributes;
insert into attributecategories values
	(1, 'Label', 2)
;


insert into attributes values 
	(1, '', 1, 1),
    (2, '', 2, 1),
    (3, '', 3, 1),
    (4, '', 4, 1),
    (5, '', 5, 1)  
; 



insert into users values ('user1', 'ユーザー１', 'user1', 'user');
insert into users values ('admin1', 'admin1', 'kyhwh6', 'administrator');


insert into datasets values (1, 'Image Classification', 'ic', 0, 0);
insert into datasets values (2, 'Object Detection', 'od', 0, 0);
insert into datasets values (3, 'Image Segmentation', 'is', 1, 1);