#!/bin/bash
# Section 4.1: Upload Processed Data to Amazon S3
# This script implements the exact command specified in the plan

set -e  # Exit on any error

echo "🚀 Section 4.1: Transferring Processed Data to the Cloud"
echo "========================================================"
echo ""

# Configuration from Section 1
S3_BUCKET="asr-finetuning-data-2025"
LOCAL_DATA_DIR="processed_data"

# Verify prerequisites
echo "🔍 Verifying prerequisites..."

# Check if AWS CLI is configured
if ! aws configure get aws_access_key_id &> /dev/null; then
    echo "❌ AWS CLI is not configured. Please run:"
    echo "   aws configure"
    echo "   OR run: ./scripts/setup_aws.sh for guidance"
    exit 1
fi

echo "✅ AWS CLI is configured"

# Check if processed data exists
if [ ! -d "$LOCAL_DATA_DIR" ]; then
    echo "❌ Processed data directory not found: $LOCAL_DATA_DIR"
    echo "   Please run Section 2 preprocessing first:"
    echo "   python src/01_preprocess.py"
    exit 1
fi

echo "✅ Processed data directory exists"

# Verify data structure
if [ ! -f "$LOCAL_DATA_DIR/train/metadata.jsonl" ] || [ ! -f "$LOCAL_DATA_DIR/test/metadata.jsonl" ]; then
    echo "❌ Missing manifest files in processed data"
    echo "   Expected: $LOCAL_DATA_DIR/train/metadata.jsonl"
    echo "   Expected: $LOCAL_DATA_DIR/test/metadata.jsonl"
    exit 1
fi

echo "✅ Manifest files found"

# Display data statistics
TRAIN_SAMPLES=$(wc -l < "$LOCAL_DATA_DIR/train/metadata.jsonl")
TEST_SAMPLES=$(wc -l < "$LOCAL_DATA_DIR/test/metadata.jsonl")
TOTAL_SIZE=$(du -sh "$LOCAL_DATA_DIR" | cut -f1)

echo ""
echo "📊 Data Summary:"
echo "   Training samples: $TRAIN_SAMPLES"
echo "   Test samples: $TEST_SAMPLES"
echo "   Total size: $TOTAL_SIZE"
echo ""

# Test S3 bucket access
echo "🧪 Testing S3 bucket access..."
if aws s3 ls "s3://$S3_BUCKET/" &> /dev/null; then
    echo "✅ Successfully connected to S3 bucket: $S3_BUCKET"
else
    echo "❌ Cannot access S3 bucket: $S3_BUCKET"
    echo "   Please check:"
    echo "   1. Bucket exists and you have access"
    echo "   2. Your AWS credentials have S3 permissions"
    echo "   3. The bucket is in the correct region"
    exit 1
fi

echo ""
echo "🔄 Starting upload to S3..."
echo "   Source: $LOCAL_DATA_DIR/"
echo "   Destination: s3://$S3_BUCKET/processed_data/"
echo ""

# Perform the upload using the exact command from Section 4.1
echo "📤 Executing: aws s3 sync $LOCAL_DATA_DIR/ s3://$S3_BUCKET/processed_data/"
echo ""

# Use --delete to remove files from S3 that are no longer in local directory
aws s3 sync "$LOCAL_DATA_DIR/" "s3://$S3_BUCKET/processed_data/" \
    --storage-class STANDARD \
    --exclude "*.DS_Store" \
    --exclude "*.log"

echo ""
echo "✅ Upload completed successfully!"
echo ""

# Verify upload
echo "🔍 Verifying upload..."
echo ""
echo "📁 S3 bucket contents:"
aws s3 ls "s3://$S3_BUCKET/processed_data/" --recursive --human-readable --summarize

echo ""
echo "🎉 Section 4.1 completed successfully!"
echo ""
echo "📋 Next Steps (Section 5):"
echo "   1. Provision cloud GPU instance (EC2)"
echo "   2. Setup CUDA and conda environment"
echo "   3. Download data from S3 to cloud VM"
echo "   4. Begin model training"
echo ""
echo "💡 The processed data is now available at:"
echo "   s3://$S3_BUCKET/processed_data/"
