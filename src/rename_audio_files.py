#!/usr/bin/env python3
"""
Audio File Renaming Script
==========================

This script renames audio files in each chapter folder so that the chapter number
in the filename matches the actual chapter folder number.

For example:
- chapter34/ch1_sent_001.wav -> chapter34/ch34_sent_001.wav
- chapter01/ch1_sent_001.wav -> chapter01/ch01_sent_001.wav

Usage:
    python src/rename_audio_files.py
"""

import os
import re
import shutil
from pathlib import Path
from typing import List, Tuple
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AudioFileRenamer:
    """Rename audio files to match their chapter folder numbers."""
    
    def __init__(self, audio_base_path: str):
        self.audio_base_path = Path(audio_base_path)
        self.backup_dir = self.audio_base_path / "backup_before_rename"
        
    def get_chapter_folders(self) -> List[Path]:
        """Get all chapter folders."""
        chapter_folders = []
        for item in self.audio_base_path.iterdir():
            if item.is_dir() and item.name.startswith("chapter"):
                chapter_folders.append(item)
        return sorted(chapter_folders)
    
    def extract_chapter_number(self, folder_name: str) -> int:
        """Extract chapter number from folder name (e.g., 'chapter34' -> 34)."""
        match = re.search(r'chapter(\d+)', folder_name)
        if match:
            return int(match.group(1))
        else:
            raise ValueError(f"Could not extract chapter number from folder name: {folder_name}")
    
    def get_files_to_rename(self, chapter_folder: Path) -> List[Tuple[Path, str]]:
        """Get list of files to rename in a chapter folder."""
        files_to_rename = []
        
        for file_path in chapter_folder.iterdir():
            if file_path.is_file() and file_path.suffix in ['.wav', '.txt']:
                # Check if filename starts with 'ch1_' (needs renaming)
                if file_path.name.startswith('ch1_'):
                    files_to_rename.append((file_path, file_path.name))
        
        return files_to_rename
    
    def create_backup(self):
        """Create a backup of the audio directory before renaming."""
        if not self.backup_dir.exists():
            logger.info(f"Creating backup at: {self.backup_dir}")
            shutil.copytree(self.audio_base_path, self.backup_dir, 
                           ignore=shutil.ignore_patterns('backup_before_rename'))
            logger.info("Backup created successfully")
        else:
            logger.info("Backup already exists, skipping backup creation")
    
    def rename_files_in_chapter(self, chapter_folder: Path) -> int:
        """Rename files in a single chapter folder."""
        chapter_number = self.extract_chapter_number(chapter_folder.name)
        files_to_rename = self.get_files_to_rename(chapter_folder)
        
        renamed_count = 0
        
        for file_path, old_name in files_to_rename:
            # Create new filename with correct chapter number
            new_name = old_name.replace('ch1_', f'ch{chapter_number:02d}_')
            new_path = file_path.parent / new_name
            
            try:
                # Rename the file
                file_path.rename(new_path)
                logger.info(f"Renamed: {old_name} -> {new_name}")
                renamed_count += 1
            except Exception as e:
                logger.error(f"Error renaming {old_name}: {e}")
        
        return renamed_count
    
    def rename_all_files(self) -> int:
        """Rename files in all chapter folders."""
        logger.info("Starting audio file renaming process...")
        
        # Create backup first
        self.create_backup()
        
        chapter_folders = self.get_chapter_folders()
        logger.info(f"Found {len(chapter_folders)} chapter folders")
        
        total_renamed = 0
        
        for chapter_folder in chapter_folders:
            logger.info(f"Processing chapter folder: {chapter_folder.name}")
            renamed_count = self.rename_files_in_chapter(chapter_folder)
            total_renamed += renamed_count
            logger.info(f"Renamed {renamed_count} files in {chapter_folder.name}")
        
        logger.info(f"Renaming complete! Total files renamed: {total_renamed}")
        return total_renamed
    
    def verify_renaming(self) -> bool:
        """Verify that all files have been renamed correctly."""
        logger.info("Verifying renaming results...")
        
        chapter_folders = self.get_chapter_folders()
        all_correct = True
        
        for chapter_folder in chapter_folders:
            chapter_number = self.extract_chapter_number(chapter_folder.name)
            expected_prefix = f'ch{chapter_number:02d}_'
            
            for file_path in chapter_folder.iterdir():
                if file_path.is_file() and file_path.suffix in ['.wav', '.txt']:
                    if not file_path.name.startswith(expected_prefix):
                        logger.error(f"File {file_path.name} in {chapter_folder.name} "
                                   f"does not have expected prefix {expected_prefix}")
                        all_correct = False
        
        if all_correct:
            logger.info("✓ All files have been renamed correctly!")
        else:
            logger.error("✗ Some files were not renamed correctly")
        
        return all_correct


def main():
    """Main function."""
    audio_base_path = "/Users/s.mengari/Documents/KAN_BASELINE/ALL_wav_txt_sentence_level_cleaned"
    
    if not os.path.exists(audio_base_path):
        logger.error(f"Audio base path does not exist: {audio_base_path}")
        return
    
    # Create renamer and execute
    renamer = AudioFileRenamer(audio_base_path)
    
    try:
        total_renamed = renamer.rename_all_files()
        
        # Verify the renaming
        if renamer.verify_renaming():
            logger.info("Audio file renaming completed successfully!")
            logger.info(f"Total files renamed: {total_renamed}")
            logger.info(f"Backup available at: {renamer.backup_dir}")
        else:
            logger.error("Renaming verification failed!")
            
    except Exception as e:
        logger.error(f"Error during renaming process: {e}")
        logger.info(f"Backup available at: {renamer.backup_dir}")


if __name__ == "__main__":
    main() 