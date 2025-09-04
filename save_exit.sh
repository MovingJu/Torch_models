#!/bin/sh
sleep 10

while true; do
  if [ ! -f $(pwd)/models/running.txt ]; then
    # git add ./models
    # git commit -m "Add models"
    # git push origin main
    sudo shutdown -h
    break
  fi
  sleep 10
done