#!/usr/bin/env python3
"""
Audio Segmentation and Preprocessing Script for Children's Speech ASR
Based on Section 2 of the implementation plan.

This script:
1. Recursively scans data/train/ and data/test/ directories
2. Parses timestamped transcription files (.txt)
3. Segments corresponding audio files (.wav) using pydub
4. Saves segments to processed_data/ directories
5. Generates metadata.jsonl manifest files for Hugging Face datasets
"""

import os
import json
import argparse
from pathlib import Path
from typing import List, Tuple, Dict
import logging

# Audio processing libraries
from pydub import AudioSegment
import librosa
import soundfile as sf

def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('preprocessing.log'),
            logging.StreamHandler()
        ]
    )

def parse_transcript_file(txt_path: str) -> List[Tuple[float, float, str]]:
    """
    Parse a transcript file and extract timestamps and text.
    
    Args:
        txt_path: Path to the .txt transcript file
        
    Returns:
        List of tuples (start_time, end_time, text)
    """
    segments = []
    
    try:
        with open(txt_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                    
                try:
                    # Split by tabs - format: start_time\tend_time\ttext
                    parts = line.split('\t')
                    if len(parts) >= 3:
                        start_time = float(parts[0])
                        end_time = float(parts[1])
                        text = '\t'.join(parts[2:])  # Join remaining parts in case text contains tabs
                        
                        # Basic validation
                        if end_time > start_time and text.strip():
                            segments.append((start_time, end_time, text.strip()))
                        else:
                            logging.warning(f"Invalid segment in {txt_path} line {line_num}: {line}")
                    else:
                        logging.warning(f"Malformed line in {txt_path} line {line_num}: {line}")
                        
                except ValueError as e:
                    logging.warning(f"Could not parse timestamps in {txt_path} line {line_num}: {line} - {e}")
                    
    except Exception as e:
        logging.error(f"Error reading {txt_path}: {e}")
        
    return segments

def segment_audio_file(wav_path: str, segments: List[Tuple[float, float, str]], 
                      output_dir: str, base_name: str) -> List[Dict[str, str]]:
    """
    Segment an audio file based on timestamps and save individual clips.
    
    Args:
        wav_path: Path to the source .wav file
        segments: List of (start_time, end_time, text) tuples
        output_dir: Directory to save segmented audio files
        base_name: Base name for output files
        
    Returns:
        List of dictionaries with file_name and transcription for manifest
    """
    manifest_entries = []
    
    try:
        # Load audio using pydub (handles various formats well)
        audio = AudioSegment.from_wav(wav_path)
        logging.info(f"Loaded audio file: {wav_path} (duration: {len(audio)/1000:.2f}s)")
        
        for i, (start_time, end_time, text) in enumerate(segments):
            try:
                # Convert seconds to milliseconds for pydub
                start_ms = int(start_time * 1000)
                end_ms = int(end_time * 1000)
                
                # Extract segment
                segment = audio[start_ms:end_ms]
                
                # Skip very short segments (less than 0.5 seconds)
                if len(segment) < 500:
                    logging.warning(f"Skipping very short segment ({len(segment)}ms): {base_name}_segment_{i}")
                    continue
                
                # Generate output filename
                output_filename = f"{base_name}_segment_{i:03d}.wav"
                output_path = os.path.join(output_dir, output_filename)
                
                # Export segment as WAV
                segment.export(output_path, format="wav")
                
                # Add to manifest
                manifest_entries.append({
                    "file_name": output_filename,
                    "transcription": text
                })
                
                logging.debug(f"Created segment: {output_filename} ({end_time-start_time:.2f}s)")
                
            except Exception as e:
                logging.error(f"Error processing segment {i} from {wav_path}: {e}")
                
    except Exception as e:
        logging.error(f"Error loading audio file {wav_path}: {e}")
        
    return manifest_entries

def find_audio_transcript_pairs(data_dir: str) -> List[Tuple[str, str]]:
    """
    Recursively find all audio-transcript pairs in the data directory.
    
    Args:
        data_dir: Root data directory to scan
        
    Returns:
        List of tuples (wav_path, txt_path)
    """
    pairs = []
    data_path = Path(data_dir)
    
    if not data_path.exists():
        logging.error(f"Data directory does not exist: {data_dir}")
        return pairs
    
    # Recursively search for directories containing both .wav and .txt files
    for subdir in data_path.rglob("*"):
        if subdir.is_dir():
            # Look for files with the same base name
            wav_files = list(subdir.glob("*.wav"))
            txt_files = list(subdir.glob("*.txt"))
            
            for wav_file in wav_files:
                base_name = wav_file.stem
                corresponding_txt = subdir / f"{base_name}.txt"
                
                if corresponding_txt in txt_files:
                    pairs.append((str(wav_file), str(corresponding_txt)))
                    logging.debug(f"Found pair: {wav_file.name} <-> {corresponding_txt.name}")
    
    logging.info(f"Found {len(pairs)} audio-transcript pairs in {data_dir}")
    return pairs

def process_dataset_split(data_dir: str, output_dir: str, split_name: str) -> int:
    """
    Process a complete dataset split (train or test).
    
    Args:
        data_dir: Input data directory
        output_dir: Output processed data directory
        split_name: Name of the split (train/test)
        
    Returns:
        Total number of segments created
    """
    logging.info(f"Processing {split_name} split: {data_dir} -> {output_dir}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all audio-transcript pairs
    pairs = find_audio_transcript_pairs(data_dir)
    
    if not pairs:
        logging.warning(f"No audio-transcript pairs found in {data_dir}")
        return 0
    
    all_manifest_entries = []
    total_segments = 0
    
    for wav_path, txt_path in pairs:
        # Parse transcript file
        segments = parse_transcript_file(txt_path)
        
        if not segments:
            logging.warning(f"No valid segments found in {txt_path}")
            continue
        
        # Extract base name for output files
        base_name = Path(wav_path).stem
        
        # Segment audio file
        manifest_entries = segment_audio_file(wav_path, segments, output_dir, base_name)
        
        all_manifest_entries.extend(manifest_entries)
        total_segments += len(manifest_entries)
        
        logging.info(f"Processed {base_name}: {len(segments)} segments -> {len(manifest_entries)} valid clips")
    
    # Write manifest file
    manifest_path = os.path.join(output_dir, "metadata.jsonl")
    with open(manifest_path, 'w', encoding='utf-8') as f:
        for entry in all_manifest_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    logging.info(f"Created manifest file: {manifest_path} with {len(all_manifest_entries)} entries")
    logging.info(f"Completed {split_name} split: {total_segments} total segments created")
    
    return total_segments

def main():
    """Main preprocessing function."""
    parser = argparse.ArgumentParser(description="Preprocess children's speech data for Whisper fine-tuning")
    parser.add_argument("--data_root", default="data/data", help="Root directory containing train/ and test/ subdirs")
    parser.add_argument("--output_root", default="processed_data", help="Output directory for processed data")
    parser.add_argument("--split", choices=["train", "test", "both"], default="both", 
                       help="Which split(s) to process")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    logging.info("Starting audio segmentation and preprocessing")
    logging.info(f"Data root: {args.data_root}")
    logging.info(f"Output root: {args.output_root}")
    
    total_segments = 0
    
    # Process train split
    if args.split in ["train", "both"]:
        train_data_dir = os.path.join(args.data_root, "train")
        train_output_dir = os.path.join(args.output_root, "train")
        total_segments += process_dataset_split(train_data_dir, train_output_dir, "train")
    
    # Process test split
    if args.split in ["test", "both"]:
        test_data_dir = os.path.join(args.data_root, "test")
        test_output_dir = os.path.join(args.output_root, "test")
        total_segments += process_dataset_split(test_data_dir, test_output_dir, "test")
    
    logging.info(f"Preprocessing completed successfully!")
    logging.info(f"Total segments created: {total_segments}")
    logging.info(f"Processed data saved to: {args.output_root}")
    logging.info("Next step: Upload processed_data/ to cloud storage for training")

if __name__ == "__main__":
    main()
