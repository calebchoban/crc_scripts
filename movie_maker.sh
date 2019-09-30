#!/bin/bash
# Script used to create movies in python

ffmpeg='/home/cchoban/packages/ffmpeg-git-20171108-64bit-static/ffmpeg'
echo "Creating Movie!"
cd $1
pwd
$ffmpeg -y -loglevel error -start_number $2 -framerate $3 -i $4 $5
