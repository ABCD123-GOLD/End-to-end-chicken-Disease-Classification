import os
import shutil
import zipfile
import py7zr
from pathlib import Path
import urllib.request as request
from Chicken_Disease_Classification.utils.logger import logger
from Chicken_Disease_Classification.utils.common import get_size
from Chicken_Disease_Classification.entity.config_entity import DataIngestionConfig
import kagglehub


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def download_from_kagglehub(self):
        try:
            # Step 1: Download to kagglehub cache first
            logger.info("Starting dataset download from Kagglehub...")
            kaggle_path = kagglehub.dataset_download(self.config.kaggle_dataset)
            logger.info(f"Dataset downloaded to Kagglehub cache: {kaggle_path}")
            
            # DIAGNOSTIC: Check what's actually in the kaggle cache
            kaggle_path_obj = Path(kaggle_path)
            logger.info(f"Kaggle cache directory exists: {kaggle_path_obj.exists()}")
            if kaggle_path_obj.exists():
                cache_items = list(kaggle_path_obj.glob("*"))
                logger.info(f"Items in Kaggle cache ({len(cache_items)} total):")
                for item in cache_items:
                    if item.is_file():
                        size_mb = item.stat().st_size / (1024*1024)
                        logger.info(f"  CACHE FILE: {item.name} ({size_mb:.2f} MB)")
                    else:
                        sub_items = list(item.glob("*"))
                        logger.info(f"  CACHE DIR:  {item.name}/ ({len(sub_items)} items)")
                        # Show a few items from subdirectories
                        for sub_item in sub_items[:3]:
                            if sub_item.is_file():
                                size_mb = sub_item.stat().st_size / (1024*1024)
                                logger.info(f"    - FILE: {sub_item.name} ({size_mb:.2f} MB)")
                            else:
                                logger.info(f"    - DIR:  {sub_item.name}/")
                        if len(sub_items) > 3:
                            logger.info(f"    ... and {len(sub_items) - 3} more items")
            
            # Step 2: Use the correct local_data_path from config
            desired_path = Path(self.config.local_data_path)
            logger.info(f"Target location: {desired_path}")
            logger.info(f"Target location (absolute): {desired_path.absolute()}")
            
            # DIAGNOSTIC: Check if target already exists and what's in it
            if desired_path.exists():
                logger.info("Target directory already exists. Contents:")
                existing_items = list(desired_path.glob("*"))
                logger.info(f"Found {len(existing_items)} items in target directory")
                for item in existing_items:
                    if item.is_file():
                        size_mb = item.stat().st_size / (1024*1024)
                        logger.info(f"  EXISTING FILE: {item.name} ({size_mb:.2f} MB)")
                    else:
                        sub_items = list(item.glob("*"))
                        logger.info(f"  EXISTING DIR:  {item.name}/ ({len(sub_items)} items)")
                
                # Check if the directory is empty or has incomplete data
                if len(existing_items) == 0:
                    logger.info("Target directory is empty - will copy data")
                    should_copy = True
                else:
                    # Check for actual data files
                    has_data = False
                    for item in existing_items:
                        if item.is_file() and item.stat().st_size > 0:
                            has_data = True
                            break
                        elif item.is_dir():
                            sub_files = list(item.rglob("*"))
                            if any(f.is_file() and f.stat().st_size > 0 for f in sub_files):
                                has_data = True
                                break
                    
                    if has_data:
                        logger.info("Target directory has valid data - skipping copy")
                        should_copy = False
                    else:
                        logger.info("Target directory exists but has no valid data - will re-copy")
                        should_copy = True
            else:
                logger.info("Target directory doesn't exist - will create and copy")
                should_copy = True
            
            # Step 3: Copy if needed
            if should_copy:
                logger.info("Copying dataset...")
                
                # Create target directory
                desired_path.mkdir(parents=True, exist_ok=True)
                
                # Clear the directory if it has incomplete data
                if desired_path.exists():
                    for item in desired_path.glob("*"):
                        if item.is_dir():
                            shutil.rmtree(item)
                            logger.info(f"Removed incomplete directory: {item.name}")
                        else:
                            item.unlink()
                            logger.info(f"Removed incomplete file: {item.name}")
                
                logger.info(f"Copying from: {kaggle_path}")
                logger.info(f"Copying to: {desired_path}")
                
                try:
                    # Copy contents of kaggle cache to our target directory
                    for item in Path(kaggle_path).glob("*"):
                        dest_item = desired_path / item.name
                        if item.is_file():
                            shutil.copy2(item, dest_item)
                            logger.info(f"Copied file: {item.name} ({item.stat().st_size / (1024*1024):.2f} MB)")
                        else:
                            shutil.copytree(item, dest_item)
                            logger.info(f"Copied directory: {item.name}")
                    
                    logger.info(f"Dataset successfully copied to: {desired_path}")
                    
                    # DIAGNOSTIC: Verify the copy worked
                    copied_items = list(desired_path.glob("*"))
                    logger.info(f"Verification: Found {len(copied_items)} items after copy")
                    
                    total_files = 0
                    total_size = 0
                    
                    for item in copied_items:
                        if item.is_file():
                            size_mb = item.stat().st_size / (1024*1024)
                            total_size += item.stat().st_size
                            total_files += 1
                            logger.info(f"  COPIED FILE: {item.name} ({size_mb:.2f} MB)")
                        else:
                            sub_files = list(item.rglob("*.jpg")) + list(item.rglob("*.jpeg")) + list(item.rglob("*.png"))
                            total_files += len(sub_files)
                            dir_size = sum(f.stat().st_size for f in item.rglob("*") if f.is_file())
                            total_size += dir_size
                            logger.info(f"  COPIED DIR:  {item.name}/ ({len(sub_files)} images, {dir_size / (1024*1024):.2f} MB)")
                    
                    logger.info(f"Copy summary: {total_files} total files, {total_size / (1024*1024):.2f} MB total")
                        
                except Exception as copy_error:
                    logger.error(f"Error during copy operation: {copy_error}")
                    logger.error(f"Source exists: {Path(kaggle_path).exists()}")
                    logger.error(f"Target exists: {desired_path.exists()}")
                    raise
                    
            else:
                logger.info(f"Using existing dataset at: {desired_path}")
                
        except Exception as e:
            logger.error(f"Error downloading/copying dataset: {e}")
            logger.error(f"Error type: {type(e).__name__}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            raise e

    def extract_zip_file(self):
        """Extract zip file or handle data.zip folder to specified directory"""
        unzip_path = Path(self.config.unzip_dir)
        data_dir = Path(self.config.local_data_path)
        
        # Debug information - let's see what's actually in the directory
        logger.info(f"Checking data directory: {data_dir.absolute()}")
        logger.info(f"Data directory exists: {data_dir.exists()}")
        
        if not data_dir.exists():
            logger.error(f"Data directory not found: {data_dir}")
            return False
        
        # List all files in the data directory
        all_items = list(data_dir.glob("*"))
        logger.info(f"Items in data directory: {len(all_items)}")
        for item in all_items:
            if item.is_file():
                size_mb = item.stat().st_size / (1024*1024)
                logger.info(f"  FILE: {item.name} ({size_mb:.2f} MB)")
            else:
                logger.info(f"  DIR:  {item.name}/")
        
        # Check specifically for data.zip folder
        data_zip_folder = data_dir / "data.zip"
        if data_zip_folder.exists() and data_zip_folder.is_dir():
            logger.info("Found 'data.zip' folder (not a zip file)")
            
            # Check what's inside the data.zip folder
            items_in_data_zip = list(data_zip_folder.glob("*"))
            logger.info(f"Items inside data.zip folder: {len(items_in_data_zip)}")
            
            if not items_in_data_zip:
                logger.warning("data.zip folder is empty!")
                logger.info("This usually means the Kaggle download didn't complete properly")
                logger.info("Try re-running the download or check your Kaggle authentication")
                return False
            
            # Show what's in the data.zip folder
            for item in items_in_data_zip:
                if item.is_file():
                    size_mb = item.stat().st_size / (1024*1024)
                    logger.info(f"  Inside data.zip - FILE: {item.name} ({size_mb:.2f} MB)")
                else:
                    # Check if subdirectories have content
                    subdir_items = list(item.glob("*"))
                    logger.info(f"  Inside data.zip - DIR: {item.name}/ ({len(subdir_items)} items)")
            
            # Copy contents of data.zip folder to unzip directory
            unzip_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Copying contents from data.zip folder to: {unzip_path.absolute()}")
            
            for item in items_in_data_zip:
                dest_path = unzip_path / item.name
                if item.is_file():
                    shutil.copy2(item, dest_path)
                    logger.info(f"Copied file: {item.name}")
                else:
                    if not dest_path.exists():
                        shutil.copytree(item, dest_path)
                        logger.info(f"Copied directory: {item.name}")
                    else:
                        logger.info(f"Directory already exists: {item.name}")
            
            # Verify the copy was successful
            copied_items = list(unzip_path.rglob('*'))
            files_count = len([item for item in copied_items if item.is_file()])
            dirs_count = len([item for item in copied_items if item.is_dir()])
            logger.info(f"Copy complete: {files_count} files, {dirs_count} directories")
            return True
        
        # Look for actual zip files
        zip_files = [item for item in all_items if item.is_file() and item.suffix.lower() == '.zip']
        if not zip_files:
            logger.warning("No .zip files found in the directory")
            logger.info("Checking if data is already extracted in other folders...")
            
            # Check if there are other folders with data
            folders = [item for item in all_items if item.is_dir() and item.name != "data.zip"]
            if folders:
                logger.info("Found other directories - copying them...")
                
                unzip_path.mkdir(parents=True, exist_ok=True)
                for folder in folders:
                    dest_folder = unzip_path / folder.name
                    if not dest_folder.exists():
                        shutil.copytree(folder, dest_folder)
                        logger.info(f"Copied: {folder.name}")
                    else:
                        logger.info(f"Already exists: {folder.name}")
                return True
            else:
                logger.error("No zip files or directories with data found")
                return False
        
        # Use the first zip file found
        zip_path = zip_files[0]
        logger.info(f"Using ZIP file: {zip_path.name}")
        
        file_size = zip_path.stat().st_size
        logger.info(f"ZIP file size: {file_size} bytes ({file_size / (1024*1024):.2f} MB)")
        
        # Create unzip directory if it doesn't exist
        unzip_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Extraction target directory: {unzip_path.absolute()}")
        
        try:
            # Check if file is actually a zip file
            if not zipfile.is_zipfile(zip_path):
                logger.error(f"File {zip_path.name} is not a valid ZIP file")
                return False
            
            logger.info("Starting ZIP extraction...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                # Get list of files in the zip
                file_list = zip_ref.namelist()
                logger.info(f"ZIP contains {len(file_list)} files/folders")
                
                # Extract all files
                zip_ref.extractall(unzip_path)
            
            logger.info(f"Successfully extracted to: {unzip_path.absolute()}")
            
            # List extracted contents for verification
            extracted_items = list(unzip_path.rglob('*'))
            files_count = len([item for item in extracted_items if item.is_file()])
            dirs_count = len([item for item in extracted_items if item.is_dir()])
            
            logger.info(f"Extraction complete: {files_count} files, {dirs_count} directories")
            
            # Show first few items for verification
            if extracted_items:
                logger.info("First few extracted items:")
                for item in extracted_items[:10]:  # Show first 10 items
                    logger.info(f"  - {item.relative_to(unzip_path)}")
                if len(extracted_items) > 10:
                    logger.info(f"  ... and {len(extracted_items) - 10} more items")
            
            return True
            
        except PermissionError as e:
            logger.error(f"Permission error: {e}")
            logger.error("Try running as administrator or check if file is being used by another process")
            raise
        except zipfile.BadZipFile as e:
            logger.error(f"Bad ZIP file error: {e}")
            logger.error("The file appears to be corrupted or not a valid ZIP file")
            raise
        except Exception as e:
            logger.error(f"Error during extraction: {e}")
            raise

    def initiate_data_ingestion(self):
        """Main method to orchestrate the data ingestion process"""
        try:
            logger.info("=" * 50)
            logger.info("Starting data ingestion process...")
            
            # Step 1: Download/copy the data from Kagglehub
            self.download_from_kagglehub()
            
            # Step 2: Extract the ZIP file
            logger.info("Starting ZIP extraction phase...")
            success = self.extract_zip_file()
            
            if success:
                logger.info("Data ingestion completed successfully!")
            else:
                logger.error("Data ingestion failed during extraction")
                return False
                
            logger.info("=" * 50)
            return True
            
        except Exception as e:
            logger.error(f"Data ingestion failed: {e}")
            raise e