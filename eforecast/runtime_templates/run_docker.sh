#!/usr/bin/env bash
docker run -v $1/nwp:/nwp -v $1/models/:/models/ -p 443:443 joesider9/ppc_demo:latest
