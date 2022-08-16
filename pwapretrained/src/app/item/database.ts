// Copyright (C) 2020 - 2022 APC Inc.

import Dexie from 'dexie';
import { IItem } from './interface';

export class DItem extends Dexie {
    items: Dexie.Table<IItem, number>;

    constructor() {
        super('DItem');
        // status 0: open camera file   1: open album file
        this.version(5).stores({
            items: 'id,filename,filetimestamp,arrayBuffer,contentType,filetype,status'
        });
    }
}