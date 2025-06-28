#!/usr/bin/env python3
"""
Complete data pipeline: download and preprocess all Verilog datasets
"""

import sys
import logging
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from data.dataset_downloader import VerilogDatasetDownloader
from data.preprocessor import VerilogPreprocessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Main pipeline function"""
    logger.info("Starting complete data pipeline...")
    
    # Step 1: Download datasets
    logger.info("=" * 50)
    logger.info("STEP 1: Downloading datasets")
    logger.info("=" * 50)
    
    downloader = VerilogDatasetDownloader()
    datasets = downloader.download_all()
    
    # Step 2: Preprocess datasets
    logger.info("=" * 50)
    logger.info("STEP 2: Preprocessing datasets")
    logger.info("=" * 50)
    
    preprocessor = VerilogPreprocessor()
    train_set, val_set, test_set = preprocessor.process_all_datasets()
    
    logger.info("=" * 50)
    logger.info("PIPELINE COMPLETED SUCCESSFULLY!")
    logger.info("=" * 50)
    logger.info(f"Training set: {len(train_set)} examples")
    logger.info(f"Validation set: {len(val_set)} examples")
    logger.info(f"Test set: {len(test_set)} examples")
    
    print("\nData pipeline completed! Ready for model training.")

if __name__ == "__main__":
    main()