#!/usr/bin/env bash
# Detach the remaining queues so that no waiting script has an ancestor whose command line names another script.
cd "$(dirname "$0")"
for s in "$@"; do
  setsid nohup "./$s" > /dev/null 2>&1 < /dev/null &
  sleep 1
done
