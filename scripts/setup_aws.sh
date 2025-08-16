#!/bin/bash
# AWS Configuration Setup Script for Section 4
# This script helps configure AWS CLI for uploading processed data to S3

echo "🔧 AWS CLI Configuration Setup for Section 4"
echo "============================================="
echo ""

# Check if AWS CLI is installed
if ! command -v aws &> /dev/null; then
    echo "❌ AWS CLI is not installed. Please install it first:"
    echo "   brew install awscli"
    echo "   OR"
    echo "   pip install awscli"
    exit 1
fi

echo "✅ AWS CLI is installed: $(aws --version)"
echo ""

# Check current configuration
echo "📋 Current AWS Configuration:"
aws configure list
echo ""

# Check if already configured
if aws configure get aws_access_key_id &> /dev/null; then
    echo "✅ AWS CLI appears to be configured."
    echo ""
    echo "🧪 Testing S3 access to bucket: asr-finetuning-data-2025"
    if aws s3 ls s3://asr-finetuning-data-2025/ &> /dev/null; then
        echo "✅ Successfully connected to S3 bucket!"
        echo ""
        echo "🚀 Ready to proceed with Section 4 data upload."
    else
        echo "❌ Cannot access S3 bucket. Please check:"
        echo "   1. Bucket exists: asr-finetuning-data-2025"
        echo "   2. Your AWS credentials have S3 permissions"
        echo "   3. The bucket is in the correct region"
    fi
else
    echo "⚠️  AWS CLI is not configured with credentials."
    echo ""
    echo "📝 To configure AWS CLI, you need:"
    echo "   1. AWS Access Key ID"
    echo "   2. AWS Secret Access Key"
    echo "   3. Default region (e.g., us-east-1)"
    echo "   4. Default output format (json recommended)"
    echo ""
    echo "🔑 Run the following command to configure:"
    echo "   aws configure"
    echo ""
    echo "   OR if you have a specific profile:"
    echo "   aws configure --profile your-profile-name"
    echo ""
    echo "📚 For more information:"
    echo "   https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-quickstart.html"
fi

echo ""
echo "📊 Current processed data status:"
if [ -d "processed_data" ]; then
    echo "✅ processed_data directory exists"
    echo "   Train samples: $(wc -l < processed_data/train/metadata.jsonl 2>/dev/null || echo 'N/A')"
    echo "   Test samples: $(wc -l < processed_data/test/metadata.jsonl 2>/dev/null || echo 'N/A')"
    echo "   Total size: $(du -sh processed_data 2>/dev/null | cut -f1 || echo 'N/A')"
else
    echo "❌ processed_data directory not found"
    echo "   Please run Section 2 preprocessing first: python src/01_preprocess.py"
fi
