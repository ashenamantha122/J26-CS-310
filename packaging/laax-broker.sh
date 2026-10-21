#!/bin/sh
# /usr/bin/laax-broker -- the LAAX message broker, system Python, isolated.
exec /usr/bin/python3 -u -sE /usr/lib/laax/run_broker.py "$@"
