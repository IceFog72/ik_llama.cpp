#!/usr/bin/env bash
# Enable CUDA stream memory operations (cuStreamWaitValue32 in hardware) on the
# NVIDIA Linux driver. GeForce drivers ship with NVreg_EnableStreamMemOPs=0, and
# the CUDA driver then emulates stream waits by blocking the host thread.
# ik_llama's --moe-resident-device-join uses the hardware wait when this is on.
#
# usage: sudo scripts/nvidia-stream-memops.sh [enable|disable|check]
# The option applies when the nvidia module loads, so reboot after enable/disable.
set -euo pipefail

CONF=/etc/modprobe.d/nvidia-stream-memops.conf
LINE="options nvidia NVreg_EnableStreamMemOPs=1"
action=${1:-enable}

current() {
    grep -E '^EnableStreamMemOPs:' /proc/driver/nvidia/params 2>/dev/null | awk '{print $2}' || true
}

# The running kernel's initramfs, from /proc/cmdline (BOOT_IMAGE=/boot/vmlinuz-NAME).
initramfs_image() {
    local image name
    image=$(tr ' ' '\n' < /proc/cmdline | sed -n 's/^BOOT_IMAGE=//p')
    name=$(basename "${image:-}")
    [[ $name == vmlinuz-* ]] && echo "/boot/initramfs-${name#vmlinuz-}.img"
}

# An initramfs that loads nvidia early carries its own copy of modprobe.d.
refresh_initramfs() {
    local img
    img=$(initramfs_image) || true
    if [[ -z ${img:-} || ! -f $img ]]; then
        echo "note: running kernel's initramfs not found; if nvidia loads from initramfs, regenerate it yourself"
        return
    fi
    if ! command -v lsinitrd >/dev/null || ! lsinitrd "$img" 2>/dev/null | grep -qE 'nvidia[^/]*\.ko|modprobe\.d/nvidia'; then
        echo "$img does not load nvidia; no regeneration needed"
        return
    fi
    echo "$img loads nvidia early and must include $CONF."
    read -r -p "Regenerate it now with: dracut --force $img $(uname -r) ? [y/N] " answer
    if [[ $answer == [yY] ]]; then
        cp -a "$img" "$img.bak-memops"
        dracut --force "$img" "$(uname -r)"
        echo "regenerated $img (backup: $img.bak-memops)"
    else
        echo "skipped; run it before rebooting or the option will not apply"
    fi
}

case $action in
    check)
        echo "EnableStreamMemOPs (running driver): $(current || echo unknown)"
        if [[ -f $CONF ]]; then echo "$CONF: $(cat "$CONF")"; else echo "$CONF: absent"; fi
        ;;
    enable)
        [[ $EUID -eq 0 ]] || { echo "run with sudo" >&2; exit 1; }
        if grep -rqsE '^\s*options\s+nvidia\s+.*NVreg_EnableStreamMemOPs=' /etc/modprobe.d --exclude="$(basename "$CONF")"; then
            echo "another /etc/modprobe.d file already sets NVreg_EnableStreamMemOPs; fix it by hand:" >&2
            grep -rsnE 'NVreg_EnableStreamMemOPs' /etc/modprobe.d >&2
            exit 1
        fi
        echo "$LINE" > "$CONF"
        echo "wrote $CONF"
        refresh_initramfs
        echo "reboot, then run: $0 check   (expect EnableStreamMemOPs: 1)"
        ;;
    disable)
        [[ $EUID -eq 0 ]] || { echo "run with sudo" >&2; exit 1; }
        rm -f "$CONF"
        echo "removed $CONF"
        refresh_initramfs
        echo "reboot to apply"
        ;;
    *)
        echo "usage: $0 [enable|disable|check]" >&2
        exit 2
        ;;
esac
