#!/usr/bin/env bash
#
# Turn a bare Ubuntu server into one that can serve a model. Runs ON the server.
#
# Part 8 sections 83 and 84. This is the section that is boring to write and is the reason
# people give up, so nothing here is skipped or hidden behind a prepared image. A Deep
# Learning AMI would make this file three lines long and would teach nothing, and the
# reader who is putting this on their own company's base image cannot use those three
# lines anyway.
#
# ⛔ THE FOUR THINGS THAT GO WRONG, in the order they went wrong on a real g6.2xlarge:
#
#   1. `nvidia-smi` is not installed, and the error says "command not found" rather than
#      "you have no driver". A fresh Ubuntu image has no NVIDIA driver at all. The GPU is
#      on the PCI bus and nothing can talk to it.
#   2. The driver installs and `nvidia-smi` still fails, because the kernel module has not
#      loaded into the running kernel. It loads on the next boot. This looks like the
#      install failed and it did not.
#   3. `pip install vllm` refuses with "externally-managed-environment". Ubuntu 24.04 ships
#      PEP 668, which stops pip writing into the system Python. The fix is a virtual
#      environment, not the --break-system-packages flag the error suggests, because that
#      flag does exactly what it says on a machine you are about to depend on.
#   4. The server installs cleanly and then dies loading the model, with an AttributeError
#      about a tokenizer. That one is a dependency that moved a major version underneath
#      an unpinned requirement, and it is the only one of the four whose message points
#      at the wrong thing entirely. See step 5.
#
# Author: Roni Das
# Created: 2026-09-10

set -euo pipefail
say() { printf '\n\033[1m== %s\033[0m\n' "$*"; }

say "1. what the machine actually has"
lspci | grep -i nvidia || echo "  no NVIDIA device on the PCI bus, wrong instance type"
nvidia-smi 2>&1 | head -3 || true

say "2. the driver"
sudo apt-get update -qq
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
     ubuntu-drivers-common build-essential python3-venv python3-pip jq >/dev/null
#
# ⛔ DO NOT NAME A DRIVER VERSION. Asking for nvidia-driver-570-server on this image
# installed 570 AND pulled 580 alongside it, and a box with two driver packages is a box
# where the kernel module and the userspace library can disagree about their version.
# `ubuntu-drivers install --gpgpu` picks the one that matches this kernel and this card,
# and --gpgpu is what keeps the desktop X stack, some three hundred packages this server
# has no screen for, off the machine.
sudo ubuntu-drivers install --gpgpu
echo "  installed:"; dpkg -l | awk '/nvidia-driver-/ {print "    " $2, $3}'

say "3. load the module into the kernel that is running now"
#
# ⛔ THIS IS FAILURE 2, AND IT COST A RUN. `apt-get install` returns as soon as the package
# is unpacked, but the driver is built against the running kernel by DKMS afterwards, and
# that takes another minute or two. Call nvidia-smi in that window and it says
#
#     NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver
#
# which reads exactly like a failed install and is not one. The module was still being
# compiled. Poll instead of asking once.
#
# ⛔ AND THIS IS FAILURE 5, WHICH WAS THIS CHECK ITSELF. The loop below used to poll
# `nvidia-smi`, wait its full five minutes and then reboot. On the second launch of this
# machine it did that twice, and the driver had been working the whole time: `lsmod`
# showed all four modules loaded within seconds. `nvidia-smi` was simply not installed.
# On this image `ubuntu-drivers install --gpgpu` picks the `no-dkms` packages, which
# bring the module and the compute libraries and nothing else, and the binary lives in
# nvidia-utils-<version>-server, which nothing in that set depends on.
#
# So the check asked "is the monitoring tool present" and reported it as "is the driver
# working". Poll for the thing you actually need, which is CUDA from Python.
#
# ⛔ AND THE FIX ABOVE HAD THE SAME FAULT AS THE THING IT FIXED. It polled
# `lsmod | grep -q '^nvidia '`, and on a g5 instance lsmod printed the row
#
#     nvidia                    -2  -2
#
# for a module that was half loaded and unusable, with no /sys/module/nvidia/holders
# behind it. The row existed, the grep matched, the script walked on, and vLLM died
# later with something that looked unrelated. A row in lsmod is not a working driver
# any more than nvidia-smi being installed was. Ask for a device node, which is the
# thing CUDA actually opens.
#
# ⛔ AND THAT FIX WAS WRONG TOO, IN A WAY WORTH KEEPING ON THE PAGE. It read
# `[ -e /dev/nvidia0 ] && nvidia-smi -L`, and nvidia-smi is installed by the step
# BELOW this loop. So the check could never pass on an image that lacks it: the
# device node was there within seconds and the loop still spent five minutes
# waiting for a binary the script had not installed yet. A readiness check must not
# depend on anything the script installs after it. /dev/nvidia0 is the signal, and
# step 5 confirms the driver properly with torch once there is a Python to ask.
ready() { [ -e /dev/nvidia0 ]; }
for i in $(seq 1 60); do
  sudo modprobe nvidia 2>/dev/null || true
  if ready; then break; fi
  [ "$i" = 1 ] && echo "  no /dev/nvidia0 yet, the driver is still coming up"
  sleep 5
done
if ! ready; then
  echo "  ⛔ no working driver after five minutes. /dev/nvidia0 is the check:"
  ls -l /dev/nvidia* 2>&1 | head -3
  lsmod | grep -i nvidia || true
  exit 1
fi
echo "  driver up, /dev/nvidia0 exists"

# ⛔ INSTALL nvidia-smi ON PURPOSE, AND TAKE THE VERSION FROM THE MACHINE. Naming a
# version here is the same mistake as naming a driver version above.
VER="$(dpkg -l | awk '/^ii +nvidia-kernel-common-[0-9]+-server/ {print $2}' \
        | sed 's/[^0-9]*\([0-9]\+\).*/\1/' | head -1)"
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
     "nvidia-utils-${VER}-server" >/dev/null
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

say "4. the python environment, because Ubuntu 24.04 will not let pip near the system one"
python3 -m venv ~/vllm-env
~/vllm-env/bin/pip install -q --upgrade pip wheel

say "5. vLLM"
# Pinned. An unpinned install of a project moving this fast means the commands in the
# article stop matching the server within weeks.
#
# ⛔ AND PIN transformers TOO, WHICH IS FAILURE 4 AND THE ONLY ONE THAT LOOKED LIKE A
# BROKEN MODEL. vLLM 0.11.0 asks for transformers>=4.55 with no upper bound, pip happily
# installed 5.17.0, and the server died on load with
#
#     AttributeError: Qwen2Tokenizer has no attribute all_special_tokens_extended
#
# That attribute was removed in transformers 5. Nothing in the message mentions
# transformers, the traceback is inside vLLM, and the natural reading is that the model
# is wrong. It is not: the model is fine and a dependency moved a major version.
~/vllm-env/bin/pip install -q "vllm==0.11.0" "transformers<5"
~/vllm-env/bin/python -c "import vllm, torch; \
print('vllm', vllm.__version__); \
print('torch', torch.__version__, 'cuda', torch.version.cuda); \
print('gpu', torch.cuda.get_device_name(0)); \
print('gpu memory GiB', round(torch.cuda.get_device_properties(0).total_memory/1024**3, 1))"

say "setup done"
