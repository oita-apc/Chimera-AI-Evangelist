// Copyright (C) 2020 - 2022 APC Inc.

import { DItem } from './database';
import { IItem } from './interface';

export class CItem {

    private dItem = new DItem();

    constructor() {
        if (!this.dItem) {
            this.dItem = new DItem();
        }
    }

    async updateStatusAll(fileType: number, status: number): Promise<void> {
        await this.dItem.items.where('filetype').equals(fileType).modify({status});
    }

    async updateStatus(item: IItem, status: number): Promise<void> {
        await this.dItem.items.update(item.id, {status});
    }

    async createItem(file: File, fileType: number, contentType: string): Promise<void> {
        // too bad in ios, we can not just save file as blob in indexedDb
        const reader = new FileReader();
        reader.onloadend = (async (evt) => {
            if (evt.target.readyState === FileReader.DONE) {
                const arrayBuffer = evt.target.result as ArrayBuffer;
                const id = await this.dItem.items.put(
                    {
                        id: Date.now(),
                        filename: file.name, filetimestamp: file.lastModified + '', arrayBuffer, contentType,
                        filetype: fileType, status: 1
                    });
                console.log('createItem id:' + id);
            }
        });
        reader.readAsArrayBuffer(file);
    }

    async getItem(fileType: number, status: number): Promise<IItem>  {
        const items = await this.dItem.items.where('filetype').equals(fileType)
            .and(x => x.status === status);
        if (await items.count() === 0) {
            return null;
        }
        const sortedItems = await items.sortBy('filetimestamp');
        return sortedItems[0];
    }

    async deleteItem(item: IItem): Promise<void> {
        await this.dItem.items.where('id').equals(item.id).delete();
    }

    async getAllItem(fileType: number): Promise<IItem[]> {
        const items = await this.dItem.items.where('filetype').equals(fileType).toArray();
        return items;
    }

    async count(fileType: number): Promise<number> {
        const count = await this.dItem.items.where('filetype').equals(fileType).count();
        console.log('filetype:' + fileType + ' count:' + count);
        return count;
    }
}
