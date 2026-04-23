#!/bin/bash
export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

if [ "$ENABLE_SSH" != "1" ]; then
    echo "SSH disabled (ENABLE_SSH != 1). Sleeping."
    exec sleep infinity
fi

mkdir -p /root/.ssh
chmod 700 /root/.ssh

if [ -n "$SSH_PUBLIC_KEY" ]; then
    echo "$SSH_PUBLIC_KEY" > /root/.ssh/authorized_keys
    chmod 600 /root/.ssh/authorized_keys
fi

mkdir -p /run/sshd
ssh-keygen -A

exec /usr/sbin/sshd -D -e
