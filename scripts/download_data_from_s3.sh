#!/bin/bash
# Section 5.3: Downloading Processed Data to the Cloud VM
# Downloads preprocessed data from S3 to the EC2 instance as specified in the plan

set -e  # Exit on any error

echo "📥 Section 5.3: Downloading Processed Data to the Cloud VM"
echo "=========================================================="
echo ""

# Configuration from Section 1
S3_BUCKET="asr-finetuning-data-2025"
LOCAL_DATA_DIR="processed_data"

# Check if AWS CLI is configured
echo "🔍 Verifying AWS CLI configuration..."
if ! aws configure get aws_access_key_id &> /dev/null; then
    echo "❌ AWS CLI is not configured. Please run:"
    echo "   aws configure"
    exit 1
fi

echo "✅ AWS CLI is configured"
echo ""

# Test S3 bucket access
echo "🧪 Testing S3 bucket access..."
if ! aws s3 ls "s3://$S3_BUCKET/" &> /dev/null; then
    echo "❌ Cannot access S3 bucket: $S3_BUCKET"
    echo "   Please check your AWS permissions"
    exit 1
fi

echo "✅ S3 bucket access confirmed"
echo ""

# Show what will be downloaded
echo "📋 S3 bucket contents to download:"
echo "---------------------------------"
aws s3 ls "s3://$S3_BUCKET/processed_data/" --recursive --human-readable --summarize

echo ""
echo "📥 Downloading processed data from S3..."
echo "---------------------------------------"
echo "Command from plan: aws s3 sync s3://asr-finetuning-data-2025/processed_data/ ./processed_data/"
echo ""

# Create local directory if it doesn't exist
mkdir -p "$LOCAL_DATA_DIR"

# Download data using the exact command from Section 5.3
aws s3 sync "s3://$S3_BUCKET/processed_data/" "./$LOCAL_DATA_DIR/"

echo ""
echo "✅ Download completed successfully!"
echo ""

# Verify downloaded data
echo "🔍 Verifying downloaded data..."
echo "-----------------------------"

if [[ -d "$LOCAL_DATA_DIR/train" && -d "$LOCAL_DATA_DIR/test" ]]; then
    echo "✅ Train and test directories found"
    
    # Count files
    train_audio_count=$(find "$LOCAL_DATA_DIR/train" -name "*.wav" | wc -l)
    test_audio_count=$(find "$LOCAL_DATA_DIR/test" -name "*.wav" | wc -l)
    
    echo "📊 Data statistics:"
    echo "  • Training audio files: $train_audio_count"
    echo "  • Test audio files: $test_audio_count"
    
    # Check manifest files
    if [[ -f "$LOCAL_DATA_DIR/train/metadata.jsonl" ]]; then
        train_manifest_lines=$(wc -l < "$LOCAL_DATA_DIR/train/metadata.jsonl")
        echo "  • Training manifest entries: $train_manifest_lines"
    fi
    
    if [[ -f "$LOCAL_DATA_DIR/test/metadata.jsonl" ]]; then
        test_manifest_lines=$(wc -l < "$LOCAL_DATA_DIR/test/metadata.jsonl")
        echo "  • Test manifest entries: $test_manifest_lines"
    fi
    
    # Calculate total size
    total_size=$(du -sh "$LOCAL_DATA_DIR" | cut -f1)
    echo "  • Total size: $total_size"
    
else
    echo "❌ Expected train/test directories not found"
    echo "   Please check the download process"
    exit 1
fi

echo ""
echo "🎯 Processed data successfully downloaded to cloud VM!"
echo "   Data is ready for training in: $LOCAL_DATA_DIR/"
echo ""
echo "📋 Section 5 Complete - Cloud GPU Environment Ready!"
echo "   Next: Section 6 - Configure Whisper Fine-Tuning Pipeline"
