#!/bin/bash
export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

# Symlink data directory if DATA_DIR differs from default
if [ -n "$DATA_DIR" ] && [ "$DATA_DIR" != "/opt/app/data" ]; then
    ln -sfn "$DATA_DIR" /opt/app/data
fi

cd /opt/app
exec python3 /opt/app/inference_app.py
