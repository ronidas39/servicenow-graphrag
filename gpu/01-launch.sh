#!/usr/bin/env bash
#
# Rent one GPU server, reachable only from this laptop, that terminates itself.
#
# Part 8 sections 81, 82 and 82b. Everything here is one command because the alternative
# is a console walkthrough that goes stale the week a tab moves, and because a server you
# created by clicking is a server you will forget to delete.
#
# ⛔ THE INSTANCE TERMINATES ITSELF. Two independent mechanisms, because one is not enough:
# `--instance-initiated-shutdown-behavior terminate` means a shutdown from inside destroys
# the machine rather than parking it, and the boot script schedules that shutdown for
# BUDGET_HOURS from now. If your laptop dies, if your session drops, if you simply forget,
# the bill still stops. A GPU left running overnight costs more than the whole article.
#
# ⛔ IT IS REACHABLE ONLY FROM THIS ADDRESS. The security group is written against the
# public IP this script sees, not 0.0.0.0/0. An open vLLM port is an open language model
# that anybody can bill you for, and they are found in minutes.
#
# Author: Roni Das
# Created: 2026-09-10

set -euo pipefail

REGION="${REGION:-us-east-1}"
INSTANCE_TYPE="${INSTANCE_TYPE:-g6.2xlarge}"
NAME="${NAME:-fcc-graphrag-gpu}"
KEY_NAME="${KEY_NAME:-fcc-graphrag-gpu-key}"
SG_NAME="${SG_NAME:-fcc-graphrag-gpu-sg}"
DISK_GB="${DISK_GB:-200}"
BUDGET_HOURS="${BUDGET_HOURS:-4}"

HERE="$(cd "$(dirname "$0")" && pwd)"
KEY_FILE="$HOME/.ssh/${KEY_NAME}.pem"
STATE="$HERE/.state"
mkdir -p "$STATE"

say() { printf '\n\033[1m%s\033[0m\n' "$*"; }

# ── the address that is allowed in ────────────────────────────────────────────────
MY_IP="$(curl -s https://checkip.amazonaws.com | tr -d '[:space:]')"
[ -n "$MY_IP" ] || { echo "could not determine this laptop's public IP"; exit 1; }
say "this laptop is $MY_IP, and it will be the only address allowed in"

# ── the key pair ──────────────────────────────────────────────────────────────────
#
# ⛔ AWS HANDS YOU THE PRIVATE KEY ONCE. There is no second copy and no recovery. If the
# file is lost the only way back into the server is to destroy it, so this writes the key
# before anything else and refuses to overwrite one that already exists.
if aws ec2 describe-key-pairs --region "$REGION" --key-names "$KEY_NAME" >/dev/null 2>&1; then
  say "key pair $KEY_NAME already exists"
  [ -f "$KEY_FILE" ] || { echo "⛔ the key pair exists in AWS but $KEY_FILE is gone."; \
    echo "   Delete the key pair and rerun, or you cannot log in."; exit 1; }
else
  say "creating key pair $KEY_NAME"
  aws ec2 create-key-pair --region "$REGION" --key-name "$KEY_NAME" \
      --query 'KeyMaterial' --output text > "$KEY_FILE"
  chmod 400 "$KEY_FILE"
  echo "  private key written to $KEY_FILE, mode 400"
fi

# ── the security group ────────────────────────────────────────────────────────────
VPC_ID="$(aws ec2 describe-vpcs --region "$REGION" --filters Name=isDefault,Values=true \
          --query 'Vpcs[0].VpcId' --output text)"
if ! SG_ID="$(aws ec2 describe-security-groups --region "$REGION" \
        --filters "Name=group-name,Values=$SG_NAME" "Name=vpc-id,Values=$VPC_ID" \
        --query 'SecurityGroups[0].GroupId' --output text 2>/dev/null)" \
     || [ "$SG_ID" = "None" ]; then
  say "creating security group $SG_NAME in $VPC_ID"
  SG_ID="$(aws ec2 create-security-group --region "$REGION" --group-name "$SG_NAME" \
           --description "fCC GraphRAG article: one laptop, one GPU server" \
           --vpc-id "$VPC_ID" --query 'GroupId' --output text)"
else
  say "security group $SG_NAME already exists as $SG_ID"
fi

# Re-authorise every time, because a home IP changes and a stale rule locks you out of
# your own server while leaving yesterday's cafe wifi allowed in.
for PORT in 22 8000 8001; do
  aws ec2 authorize-security-group-ingress --region "$REGION" --group-id "$SG_ID" \
      --protocol tcp --port "$PORT" --cidr "${MY_IP}/32" >/dev/null 2>&1 \
    && echo "  opened $PORT to ${MY_IP}/32" \
    || echo "  $PORT already open to ${MY_IP}/32"
done

# ── the boot script ───────────────────────────────────────────────────────────────
BOOT="$(mktemp)"
cat > "$BOOT" <<BOOTEOF
#!/bin/bash
# The safety net, armed before anything else runs. Paired with
# --instance-initiated-shutdown-behavior terminate, this destroys the machine.
shutdown -h +$(( BUDGET_HOURS * 60 )) "budget reached, terminating" &
echo "auto-terminate armed for +${BUDGET_HOURS}h at \$(date -u)" > /var/log/fcc-budget.log
BOOTEOF

AMI_ID="$(aws ec2 describe-images --region "$REGION" --owners 099720109477 \
  --filters 'Name=name,Values=ubuntu/images/hvm-ssd-gp3/ubuntu-noble-24.04-amd64-server-*' \
            'Name=state,Values=available' \
  --query 'sort_by(Images,&CreationDate)[-1].ImageId' --output text)"

PRICE="$(aws pricing get-products --region us-east-1 --service-code AmazonEC2 \
  --filters "Type=TERM_MATCH,Field=instanceType,Value=$INSTANCE_TYPE" \
            "Type=TERM_MATCH,Field=location,Value=US East (N. Virginia)" \
            "Type=TERM_MATCH,Field=operatingSystem,Value=Linux" \
            "Type=TERM_MATCH,Field=tenancy,Value=Shared" \
            "Type=TERM_MATCH,Field=preInstalledSw,Value=NA" \
            "Type=TERM_MATCH,Field=capacitystatus,Value=Used" --output json \
  | python3 -c "import json,sys;d=json.load(sys.stdin);o=json.loads(d['PriceList'][0]);\
print(list(list(o['terms']['OnDemand'].values())[0]['priceDimensions'].values())[0]['pricePerUnit']['USD'])")"

say "launching one $INSTANCE_TYPE from $AMI_ID"
echo "  on demand \$$PRICE an hour, budget ${BUDGET_HOURS}h, ceiling \
\$$(python3 -c "print(f'{float('$PRICE')*$BUDGET_HOURS:.2f}')")"

INSTANCE_ID="$(aws ec2 run-instances --region "$REGION" \
  --image-id "$AMI_ID" --instance-type "$INSTANCE_TYPE" \
  --key-name "$KEY_NAME" --security-group-ids "$SG_ID" \
  --instance-initiated-shutdown-behavior terminate \
  --block-device-mappings "[{\"DeviceName\":\"/dev/sda1\",\"Ebs\":{\"VolumeSize\":$DISK_GB,\"VolumeType\":\"gp3\",\"DeleteOnTermination\":true}}]" \
  --user-data "file://$BOOT" \
  --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=$NAME},{Key=Project,Value=fcc-servicenow-graphrag}]" \
  --query 'Instances[0].InstanceId' --output text)"
rm -f "$BOOT"

echo "  $INSTANCE_ID"
aws ec2 wait instance-running --region "$REGION" --instance-ids "$INSTANCE_ID"
PUBLIC_IP="$(aws ec2 describe-instances --region "$REGION" --instance-ids "$INSTANCE_ID" \
             --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)"

cat > "$STATE/instance.env" <<EOF
REGION=$REGION
INSTANCE_ID=$INSTANCE_ID
INSTANCE_TYPE=$INSTANCE_TYPE
PUBLIC_IP=$PUBLIC_IP
KEY_FILE=$KEY_FILE
SG_ID=$SG_ID
KEY_NAME=$KEY_NAME
SG_NAME=$SG_NAME
PRICE_PER_HOUR=$PRICE
LAUNCHED_AT=$(date -u +%Y-%m-%dT%H:%M:%SZ)
BUDGET_HOURS=$BUDGET_HOURS
EOF

say "running at $PUBLIC_IP"
echo "  ssh -i $KEY_FILE ubuntu@$PUBLIC_IP"
echo "  state written to $STATE/instance.env"
echo "  ⛔ it terminates itself in ${BUDGET_HOURS}h. Run 05-teardown.sh when you finish."
