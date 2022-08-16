#!/bin/bash
# Copyright (C) 2020 - 2022 APC, Inc.

echo "BASE_DIR: $BASE_DIR"
cd $BASE_DIR/mlapp/gatewayapp 
pwd 
rm ./logs/lock
python -u main.py  | /usr/bin/multilog t s1048576 n100 ./logs
# TODO: implement restart with pipe
#| /usr/bin/multilog t s1048576 n100 ./logs