#!/usr/bin/env bash
#
# Destroy everything the GPU work created, and prove there is nothing left billing.
#
# Part 8 section 88. This is the section people skip and it is the one that costs money.
# Stopping an instance stops the compute charge and keeps charging for the disk. Deleting
# an instance without releasing its elastic address keeps charging for the address. The
# only safe end state is that a query for anything tagged with this project returns
# nothing, and that is what this script checks rather than asserts.
#
# ⛔ IT TOUCHES ONLY WHAT IT MADE. Every delete is filtered on the project tag or on the
# exact names 01-launch.sh created. There are other instances in this account that belong
# to other work and nothing here can reach them.
#
# Author: Roni Das
# Created: 2026-09-10

set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
STATE="$HERE/.state/instance.env"
KEEP_KEY="${KEEP_KEY:-0}"

say() { printf '\n\033[1m%s\033[0m\n' "$*"; }

[ -f "$STATE" ] || { echo "no $STATE, nothing was launched from here"; exit 0; }
# shellcheck disable=SC1090
source "$STATE"

say "1. what is running under this project's tag"
aws ec2 describe-instances --region "$REGION" \
  --filters "Name=tag:Project,Values=fcc-servicenow-graphrag" \
            "Name=instance-state-name,Values=pending,running,stopping,stopped" \
  --query 'Reservations[].Instances[].{Id:InstanceId,Type:InstanceType,State:State.Name}' \
  --output table

say "2. terminating $INSTANCE_ID"
aws ec2 terminate-instances --region "$REGION" --instance-ids "$INSTANCE_ID" \
  --query 'TerminatingInstances[].{Id:InstanceId,From:PreviousState.Name,To:CurrentState.Name}' \
  --output table
aws ec2 wait instance-terminated --region "$REGION" --instance-ids "$INSTANCE_ID"
echo "  terminated"

say "3. the disk"
# DeleteOnTermination was set at launch, so this is a check and not a delete. A volume
# that survived its instance is the single most common forgotten charge in an AWS account.
LEFT="$(aws ec2 describe-volumes --region "$REGION" \
  --filters "Name=status,Values=available" \
  --query "Volumes[?Tags[?Key=='Project'&&Value=='fcc-servicenow-graphrag']].VolumeId" \
  --output text)"
[ -z "$LEFT" ] && echo "  no orphaned volumes" || { echo "  orphaned: $LEFT"; \
  for v in $LEFT; do aws ec2 delete-volume --region "$REGION" --volume-id "$v"; \
  echo "    deleted $v"; done; }

say "4. elastic addresses"
# This project never allocated one. The check stays because an address allocated and not
# associated is charged by the hour, and it is invisible on the instances page.
ADDRS="$(aws ec2 describe-addresses --region "$REGION" \
  --query "Addresses[?Tags[?Key=='Project'&&Value=='fcc-servicenow-graphrag']].AllocationId" \
  --output text)"
[ -z "$ADDRS" ] && echo "  none allocated by this project" || { \
  for a in $ADDRS; do aws ec2 release-address --region "$REGION" --allocation-id "$a"; \
  echo "  released $a"; done; }

say "5. the security group"
aws ec2 delete-security-group --region "$REGION" --group-id "$SG_ID" 2>/dev/null \
  && echo "  deleted $SG_NAME" || echo "  $SG_NAME still in use or already gone"

say "6. the key pair"
if [ "$KEEP_KEY" = "1" ]; then
  echo "  kept $KEY_NAME, KEEP_KEY=1"
else
  aws ec2 delete-key-pair --region "$REGION" --key-name "$KEY_NAME" && \
    echo "  deleted $KEY_NAME from AWS"
  rm -f "$KEY_FILE" && echo "  deleted $KEY_FILE"
fi

say "7. proof, not a claim"
REMAIN="$(aws ec2 describe-instances --region "$REGION" \
  --filters "Name=tag:Project,Values=fcc-servicenow-graphrag" \
            "Name=instance-state-name,Values=pending,running,stopping,stopped" \
  --query 'length(Reservations[].Instances[])' --output text)"
echo "  instances still billable under this project: $REMAIN"
[ "$REMAIN" = "0" ] || { echo "  ⛔ something is still running"; exit 1; }

RAN_FOR="$(python3 - "$LAUNCHED_AT" <<'PY'
import datetime, sys
t0 = datetime.datetime.strptime(sys.argv[1], "%Y-%m-%dT%H:%M:%SZ").replace(
    tzinfo=datetime.timezone.utc)
now = datetime.datetime.now(datetime.timezone.utc)
print(round((now - t0).total_seconds() / 3600, 2))
PY
)"
echo "  $INSTANCE_TYPE ran for ${RAN_FOR}h at \$$PRICE_PER_HOUR an hour"
echo "  compute cost \$$(python3 -c "print(f'{$RAN_FOR*float('$PRICE_PER_HOUR'):.2f}')")"
mv "$STATE" "$STATE.done"
say "nothing left running"
