#!/bin/bash
/opt/homebrew/bin/mogrify -auto-orient $2  &&  /opt/homebrew/bin/convert -thumbnail $1 $2 $3 && /opt/homebrew/bin/magick identify $2
