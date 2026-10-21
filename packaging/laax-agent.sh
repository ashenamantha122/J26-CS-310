#!/bin/sh
# /usr/bin/laax-agent -- runs the packaged agent with the system Python.
#   -u  unbuffered, so every line reaches the journal as it is written
#   -s  ignore the user site-packages directory
#   -E  ignore PYTHON* environment variables
# This runs as root; neither a user's site-packages nor a stray PYTHONPATH may
# decide which code gets loaded.
exec /usr/bin/python3 -u -sE /usr/lib/laax/laax_agent.py "$@"
