#!/bin/bash
# Push local code to the cluster (code only; logs stay on the cluster).
set -euo pipefail
cd "$(dirname "$0")"
rsync -az --exclude logs/ --exclude .git/ --exclude __pycache__/ \
  -e "sshpass -f $HOME/.ssh/.ssh_password ssh" ./ qx235@unicorn-login-01.coecis.cornell.edu:unicorn/
