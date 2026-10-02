"""
Dataset Downloader Module
Handles automated downloading and preparation of test imagery and navigation benchmarks.
"""

from pathlib import Path
import urllib.request


class RealDatasetDownloader:
    """Download real traffic and pedestrian navigation dataset."""
    
    def __init__(self):
        self.dataset_urls = [
            "http://farm4.staticflickr.com/3666/10277303256_a6a11a9d4b_z.jpg",
            "http://farm3.staticflickr.com/2538/4236286875_05b2e96ec4_z.jpg",
            "http://farm7.staticflickr.com/6141/6022871891_a601326786_z.jpg",
            "http://images.cocodataset.org/val2017/000000001268.jpg",
            "http://farm9.staticflickr.com/8118/8965896602_c68fe611bd_z.jpg",
            "http://farm8.staticflickr.com/7073/7345527746_6b25ae7ac1_z.jpg",
            "http://images.cocodataset.org/val2017/000000000724.jpg",
            "http://images.cocodataset.org/val2017/000000001584.jpg",
            "http://images.cocodataset.org/val2017/000000002006.jpg",
        ]
    
    def download_dataset(self, output_dir):
        """Download evaluation dataset into target directory."""
        print("\n" + "=" * 80)
        print("DOWNLOADING TRAFFIC & PEDESTRIAN NAVIGATION DATASET")
        print("=" * 80)
        
        image_dir = Path(output_dir) / 'images'
        image_dir.mkdir(parents=True, exist_ok=True)
        
        for idx, url in enumerate(self.dataset_urls, 1):
            filename = url.split('/')[-1]
            output_path = image_dir / filename
            
            if output_path.exists():
                print(f"[{idx}/{len(self.dataset_urls)}] ✓ Already downloaded: {filename}")
                continue
            
            try:
                print(f"[{idx}/{len(self.dataset_urls)}] ⬇ Downloading: {filename}")
                urllib.request.urlretrieve(url, output_path)
                print("    ✓ Saved successfully")
            except Exception as e:
                print(f"    ✗ Error: {e}")
        
        print("\n✓ Dataset ready for processing!")
        return image_dir
