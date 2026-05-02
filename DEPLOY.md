# NextTick - AWS Deployment Guide

Deploy NextTick to AWS using EC2 + S3 + IAM + Docker. Stays within AWS Free Tier (12 months).

---

## Architecture

```
User browser
    |
    v
[EC2 t3.micro]      Flask + gunicorn in Docker
    |   IAM Role -> S3 read-only
    v
[S3 bucket]         7 model artifacts, fetched at container startup
    |
    v
[Yahoo Finance]     live OHLCV + market context
```

**Why this design:** EC2 hosts the app, S3 stores model artifacts decoupled from the image (retrain -> reupload -> restart, no rebuild), IAM role provides scoped credentials with no secrets in code, Docker keeps the environment reproducible.

---

## Cost (within Free Tier)

| Service | Free allowance | Usage | Cost |
|---|---|---|---|
| EC2 t3.micro | 750 hrs/mo | 720 hrs | $0 |
| EBS gp3 | 30 GB | ~10 GB | $0 |
| S3 Standard | 5 GB | 25 MB | $0 |
| Data transfer out | 100 GB/mo | < 1 GB | $0 |
| Elastic IP (attached) | Free | 1 | $0 |

After 12 months: ~$8.50/mo.

---

## Prerequisites

- AWS account with Free Tier eligibility
- IAM admin user (don't use root)
- AWS CLI installed locally
- Trained model artifacts in `models/` (run the notebooks first)

---

## Step 1 - S3 bucket

1. S3 console -> **Create bucket**
2. Name: `nexttick-models-<unique-suffix>` (must be globally unique)
3. Region: us-east-1
4. Block Public Access: leave all 4 boxes checked
5. Versioning: disabled
6. Encryption: SSE-S3 (default)
7. Open the bucket -> **Upload** -> add all 7 files from `models/`:
    - `logistic_regression.pkl`, `random_forest_classifier.pkl`, `linear_regression.pkl`, `random_forest_regressor.pkl`, `lstm_classifier.pt`, `lstm_regressor.pt`, `scaler.pkl`

---

## Step 2 - IAM role for EC2

1. IAM console -> Roles -> **Create role**
2. Trusted entity: AWS service. Use case: EC2. Next.
3. Skip permissions, name the role `nexttick-ec2-role`, create.
4. Open the role -> **Add permissions** -> **Create inline policy** -> JSON tab:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": ["s3:GetObject", "s3:ListBucket"],
      "Resource": [
        "arn:aws:s3:::<your-bucket-name>",
        "arn:aws:s3:::<your-bucket-name>/*"
      ]
    }
  ]
}
```

5. Name the policy `nexttick-s3-read`, create.

Least-privilege: this role can only read the one bucket.

---

## Step 3 - Launch EC2

1. EC2 console -> **Launch instance**
2. Settings:
    - Name: `nexttick-server`
    - AMI: Amazon Linux 2023
    - Instance type: t3.micro
    - Key pair: create new (`nexttick-key`, RSA, .pem). Save the file.
    - Security group: new, with inbound rules
        - SSH (22) from My IP
        - HTTP (80) from Anywhere-IPv4
    - Storage: 8-20 GiB gp3
    - Advanced details -> IAM instance profile: `nexttick-ec2-role`
3. Launch. Wait until status checks pass.
4. (Optional but recommended) Allocate an Elastic IP and associate it with the instance. Free while attached, gives a static public IP.

---

## Step 4 - SSH and deploy

Lock down the key on Windows:

```powershell
icacls <path>\nexttick-key.pem /inheritance:r
icacls <path>\nexttick-key.pem /grant:r "${env:USERNAME}:R"
```

Connect:

```powershell
ssh -i <path>\nexttick-key.pem ec2-user@<public-ip>
```

On EC2:

```bash
sudo dnf update -y
sudo dnf install -y docker git
sudo systemctl start docker
sudo systemctl enable docker
sudo usermod -aG docker ec2-user
exit
```

Reconnect for group membership to apply, then verify the IAM role:

```bash
TOKEN=$(curl -X PUT "http://169.254.169.254/latest/api/token" \
  -H "X-aws-ec2-metadata-token-ttl-seconds: 21600" -s)
curl -H "X-aws-ec2-metadata-token: $TOKEN" \
  http://169.254.169.254/latest/meta-data/iam/security-credentials/
# Expected: nexttick-ec2-role
```

Build and run:

```bash
git clone https://github.com/<your-github>/NextTick.git
cd NextTick/flask_app

docker build -t nexttick:latest .

docker run -d \
  --name nexttick \
  --restart unless-stopped \
  -p 80:5000 \
  -e NEXTTICK_S3_BUCKET=<your-bucket-name> \
  -e AWS_DEFAULT_REGION=us-east-1 \
  nexttick:latest

docker logs -f nexttick
```

Expected logs: `Found credentials from IAM Role`, 7 download lines, `Artifact status: {... 'lstm_cls': 'ok', 'lstm_reg': 'ok'}`, `Listening at: http://0.0.0.0:5000`.

Test in browser: `http://<public-ip>`.

---

## Step 5 - Free domain (optional)

1. https://www.duckdns.org/ -> sign in with GitHub
2. Add a subdomain (e.g. `nexttick`)
3. Set its IP to the Elastic IP -> update
4. URL becomes `http://nexttick.duckdns.org`

---

## Updating the deployment

### Code change

```bash
# Local
git push origin master

# EC2
cd ~/NextTick
git fetch origin master && git reset --hard origin/master
cd flask_app
docker build -t nexttick:latest .
docker stop nexttick && docker rm nexttick
docker run -d --name nexttick --restart unless-stopped -p 80:5000 \
  -e NEXTTICK_S3_BUCKET=<your-bucket-name> \
  -e AWS_DEFAULT_REGION=us-east-1 \
  nexttick:latest
```

### Model retrained

Models live in S3, not the image. Re-upload and restart the container:

```bash
aws s3 cp models/random_forest_classifier.pkl s3://<your-bucket-name>/
# On EC2:
docker restart nexttick
```

### Deploy script

Save on EC2 as `~/deploy.sh`:

```bash
#!/bin/bash
set -e
cd ~/NextTick
git fetch origin master && git reset --hard origin/master
cd flask_app
docker build -t nexttick:latest .
docker stop nexttick || true
docker rm nexttick || true
docker run -d --name nexttick --restart unless-stopped -p 80:5000 \
  -e NEXTTICK_S3_BUCKET=<your-bucket-name> \
  -e AWS_DEFAULT_REGION=us-east-1 \
  nexttick:latest
docker logs --tail 20 nexttick
```

`chmod +x ~/deploy.sh`. Future deploys: `~/deploy.sh`.

---
