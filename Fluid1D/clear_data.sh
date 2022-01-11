#!/bin/sh

if rm ./data/* 2>/dev/null ; then echo "Data folder cleared."
else echo "Data folder already empty."
fi