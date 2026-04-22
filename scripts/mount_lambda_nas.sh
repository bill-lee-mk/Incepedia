#!/usr/bin/env bash
# Incepedia · Lambda Labs shared-storage attach helper
#
# CONTEXT
# -------
# On Lambda Labs cloud (e.g. our 192-222-52-165 H100 host), the NAS at
# /lambda/nfs/us-south-2 is exposed via VIRTIO-FS, NOT NFS — it is attached at
# the VM hypervisor level (volume UUID = aa4b366d-e191-4f78-a42a-f817ce6b86ee
# in our case).  This means a sibling host (e.g. our 150-136-213-164 A100 box
# used for evaluation) cannot just `mount -t nfs ...` to get the same data —
# the volume must be attached *at the VM level* by Lambda.
#
# This script implements three strategies, in order of preference:
#
#   1. ATTACH (preferred) — verify the Lambda virtiofs volume is already
#      attached and persistent in /etc/fstab; if so, mount it.  Use this on
#      hosts that have had the shared volume attached via the Lambda console
#      or `lambda-cloud` CLI.
#
#   2. SSHFS (fallback) — if the volume cannot be attached, mount the NAS
#      remotely over SSH from the primary host.  Usable but slower.
#
#   3. RSYNC-only (no mount) — give up on a live mount and use rsync via SSH
#      for explicit, on-demand transfers (per-experiment, not continuous).
#      This is what scripts/sync_to_nas.sh already does today.
#
# Usage:
#   bash scripts/mount_lambda_nas.sh attach           # try strategy 1
#   bash scripts/mount_lambda_nas.sh sshfs <peer>     # strategy 2 (remote primary host)
#   bash scripts/mount_lambda_nas.sh check            # report current state
#
# Environment overrides:
#   INCEPEDIA_NAS_ROOT  — local path (default /lambda/nfs/us-south-2)
#   LAMBDA_VOLUME_UUID  — virtiofs source UUID for strategy 1
#                         (default aa4b366d-e191-4f78-a42a-f817ce6b86ee)
#   PEER_USER, PEER_HOST — used by strategy 2

set -euo pipefail

NAS="${INCEPEDIA_NAS_ROOT:-/lambda/nfs/us-south-2}"
VOL_UUID="${LAMBDA_VOLUME_UUID:-aa4b366d-e191-4f78-a42a-f817ce6b86ee}"
MODE="${1:-check}"

ok()   { echo "  ✅ $*"; }
warn() { echo "  ⚠  $*"; }
fail() { echo "  ❌ $*"; }

ensure_dir () {
  if [[ ! -d "$NAS" ]]; then
    sudo mkdir -p "$NAS"
    sudo chown "$USER":"$USER" "$NAS"
  fi
}

is_mounted () { mountpoint -q "$NAS"; }

case "$MODE" in
  check)
    echo "── current NAS state ──"
    if is_mounted; then
      ok "$NAS is mounted"
      mount | grep "$NAS" || true
      df -h "$NAS"
    else
      warn "$NAS is NOT mounted"
      [[ -d "$NAS" ]] && ls -la "$NAS" | head -3 || echo "  (directory does not exist)"
    fi
    echo
    echo "── /etc/fstab entry (if any) ──"
    grep -E " $NAS " /etc/fstab || echo "  (none)"
    echo
    echo "── lambda-related units ──"
    systemctl list-unit-files 2>/dev/null | grep -i lambda | head -5 || echo "  (none)"
    ;;

  attach)
    echo "── strategy 1: virtiofs attach ──"
    ensure_dir
    if is_mounted; then ok "already mounted"; exit 0; fi
    # Look for a virtiofs source matching our UUID.
    if ls /dev/disk/by-uuid/"$VOL_UUID" >/dev/null 2>&1 \
       || lsblk -o NAME,UUID 2>/dev/null | grep -q "$VOL_UUID" \
       || grep -q "$VOL_UUID" /proc/mounts; then
      ok "virtiofs volume $VOL_UUID is visible to the kernel"
    else
      fail "virtiofs volume $VOL_UUID not visible — Lambda must attach it via the console first"
      cat <<EOF

  Action required (one-time, on Lambda Labs side):
    1. Open Lambda Labs Cloud Console
    2. Navigate to your A100 instance (164)
    3. Go to "Storage" / "Shared Volumes"
    4. Attach existing shared filesystem with UUID:  $VOL_UUID
       (this is the same volume already attached to host 165)
    5. Reboot the A100 instance OR run:
       sudo systemctl restart lambda-nfs-us\\x2dsouth\\x2d2.mount
EOF
      exit 1
    fi
    # Add to fstab if missing (idempotent).
    if ! grep -q " $NAS " /etc/fstab; then
      LINE="$VOL_UUID	$NAS	virtiofs	async,auto,exec,nodev,nosuid,nouser,rw,comment=cloudconfig	0	0"
      echo "$LINE" | sudo tee -a /etc/fstab
      ok "added /etc/fstab line"
    fi
    sudo systemctl daemon-reload
    sudo mount "$NAS"
    is_mounted && ok "mounted successfully" || { fail "mount failed"; exit 1; }
    df -h "$NAS"
    ;;

  sshfs)
    PEER="${2:-}"
    if [[ -z "$PEER" ]]; then
      fail "usage: mount_lambda_nas.sh sshfs <user@peer-host>"
      exit 1
    fi
    echo "── strategy 2: sshfs from $PEER ──"

    # Check 0: already mounted? — early-out before doing anything else.
    if is_mounted; then
      ok "$NAS already mounted on this host (no action needed)"
      mount | grep "$NAS" || true
      exit 0
    fi

    # Check 1: are we trying to sshfs to ourselves? — silly mistake.
    SELF_HOST_NAMES="$(hostname; hostname -i 2>/dev/null; hostname -I 2>/dev/null) localhost 127.0.0.1"
    PEER_HOST="${PEER##*@}"   # strip user@ if present
    for h in $SELF_HOST_NAMES; do
      if [[ "$PEER_HOST" == "$h" ]]; then
        warn "peer '$PEER' looks like THIS host ($h)."
        warn "On Lambda H100 hosts (165) the NAS is already a virtiofs mount and"
        warn "this script is a no-op.  You probably meant to run this on the A100"
        warn "host (164) with: bash scripts/mount_lambda_nas.sh sshfs ubuntu@192.222.52.165"
        exit 0
      fi
    done

    # Now install sshfs if needed.
    if ! command -v sshfs >/dev/null; then
      echo "  Installing sshfs (sudo apt-get install -y sshfs)..."
      if sudo apt-get update -qq && sudo apt-get install -y sshfs; then
        ok "sshfs installed"
      else
        fail "sshfs install failed (apt lock?). Retry after the other apt finishes:"
        echo "    sudo lsof /var/lib/apt/lists/lock"
        echo "    # wait for the holder to release, then re-run this command."
        exit 1
      fi
    fi

    ensure_dir

    # Common sshfs options:
    #   reconnect + ServerAlive*: survive transient SSH drops
    #   IdentitiesOnly + StrictHostKeyChecking=accept-new: be explicit & safe
    SSHFS_OPTS_BASE="reconnect,ServerAliveInterval=15,ServerAliveCountMax=3,StrictHostKeyChecking=accept-new"

    # `allow_other` is only needed for multi-user access (e.g. root/other UIDs
    # reading the same mount).  For our single-user setup it is unnecessary
    # AND requires either root or `user_allow_other` in /etc/fuse.conf.
    # If you actually need it, run:
    #     echo 'user_allow_other' | sudo tee -a /etc/fuse.conf
    USE_ALLOW_OTHER="${USE_ALLOW_OTHER:-0}"
    if [[ "$USE_ALLOW_OTHER" == "1" ]]; then
      SSHFS_OPTS="$SSHFS_OPTS_BASE,allow_other"
    else
      SSHFS_OPTS="$SSHFS_OPTS_BASE"
    fi

    set +e
    sshfs -o "$SSHFS_OPTS" "$PEER:$NAS" "$NAS"
    rc=$?
    set -e
    if [[ $rc -ne 0 ]]; then
      fail "sshfs returned $rc"
      cat <<'EOF'

  Common causes:
    * 'allow_other only allowed if user_allow_other is set in /etc/fuse.conf'
        → set USE_ALLOW_OTHER=0 (default) so we skip it; OR enable it once:
          echo 'user_allow_other' | sudo tee -a /etc/fuse.conf
    * 'Permission denied (publickey)'
        → first ensure ssh-agent forwarding works:  ssh -A ubuntu@<peer>
          and that the peer accepts your key (check ~/.ssh/authorized_keys).
    * 'Connection refused' / 'No route to host'
        → check the peer IP and network reachability:  ping <peer>
EOF
      exit "$rc"
    fi

    if is_mounted; then
      ok "sshfs mount established"
    else
      fail "sshfs reported success but $NAS is not mounted; aborting"
      exit 1
    fi
    df -h "$NAS"
    echo
    warn "throughput is SSH-encryption bound (~50-200 MB/s); for bulk"
    warn "ckpt/dataset transfer prefer 'rsync' from scripts/sync_to_nas.sh."
    ;;

  unmount|umount)
    if is_mounted; then
      sudo umount "$NAS" && ok "unmounted"
    else
      ok "was not mounted"
    fi
    ;;

  *)
    sed -n '2,40p' "$0"
    exit 1
    ;;
esac
