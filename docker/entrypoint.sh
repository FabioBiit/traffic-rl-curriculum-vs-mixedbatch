#!/bin/bash
set -e

if [ "${USE_XVFB:-0}" = "1" ] && command -v Xvfb > /dev/null 2>&1; then
    exec xvfb-run -a -s "-screen 0 1280x720x24" python "$@"
else
    exec python "$@"
fi
