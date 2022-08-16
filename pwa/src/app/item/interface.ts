// Copyright (C) 2020 - 2022 APC Inc.

export interface IItem {
    id?: number;
    filename?: string;
    filetimestamp?: string;
    arrayBuffer?: ArrayBuffer;
    contentType?: string;
    filetype: number;
    status?: number;
}
