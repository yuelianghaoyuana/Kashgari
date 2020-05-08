#!/usr/bin/env bash

sh scripts/docs.sh

python3 -m http.server --directory _site
