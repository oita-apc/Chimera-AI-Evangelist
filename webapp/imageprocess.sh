#!/bin/bash
/usr/bin/mogrify -auto-orient $2  &&  /usr/bin/convert -thumbnail $1 $2 $3 && /home/ubuntu/QA/magick/magick identify $2
