import asyncio
import csv
import os
import random
import re
from datetime import datetime
from playwright.async_api import async_playwright
from google.cloud import storage

def save_data_with_append(all_data, data_dir="data", base_filename="gta6_dataset.csv"):
    """
    Save data to CSV with append functionality for batch processing
    """
    if not all_data:
        print("No data to save")
        return None
        
    csv_path = os.path.join(data_dir, base_filename)
    
    # Check if file exists to determine if we need to write headers
    file_exists = os.path.exists(csv_path)
    
    fieldnames = [
        "video_id", "video_url", "caption", "likes", 
        "comments", "saves", "raw_likes", "raw_comments", "raw_saves",
        "comment_text", "comments_data", "comment_count_extracted", "timestamp"
    ]
    
    try:
        # Open in append mode ('a') instead of write mode ('w')
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            
            # Only write header if file is new
            if not file_exists:
                writer.writeheader()
                print(f"Created new dataset file: {csv_path}")
            else:
                print(f"Appending to existing dataset: {csv_path}")
                
            writer.writerows(all_data)
        
        print(f"Added {len(all_data)} videos to dataset")
        return csv_path
    except Exception as e:
        print(f"Error saving data to CSV: {e}")
        return None

def load_existing_video_ids(data_dir="data", base_filename="gta6_dataset.csv"):
    """
    Load existing video IDs to avoid duplicates across batches
    """
    csv_path = os.path.join(data_dir, base_filename)
    existing_ids = set()
    
    if os.path.exists(csv_path):
        try:
            with open(csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row.get('video_id'):
                        existing_ids.add(row['video_id'])
            print(f"Loaded {len(existing_ids)} existing video IDs from dataset")
        except Exception as e:
            print(f"Error reading existing dataset: {e}")
    
    return existing_ids


# Configuration
HASHTAG = "gta6"  # Single hashtag to scrape
GCP_CREDENTIALS_PATH = "/Users/deepakthumma/Desktop/gcp-creds/elliptical-rite-464103-b7-97c5e8f7add2.json"
GCS_BUCKET_NAME = "tiktok-sentiment-data"
GCS_DEST_BLOB = "tiktok_gta6_data.csv"
DATA_DIR = "data"
BASE_FILENAME = "gta6_dataset.csv"  # Single filename for all data
MAX_VIDEOS_TOTAL = 200  # Total videos to scrape
MAX_COMMENTS_PER_VIDEO = 20  # Maximum comments to extract per video
SCROLL_PAUSE_RANGE = (2, 4)
RETURN_TO_HASHTAG_PAUSE_RANGE = (3, 6)  # Pause when returning to hashtag page
COMMENT_EXTRACTION_ENABLED = True  # Set to False to disable comment extraction
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
]

GTA6_KEYWORDS = ["gta6", "gta 6", "grand theft auto", "rockstar", "lucia", "vice city"]

def contains_gta6(text):
    """Check if text contains GTA6-related keywords"""
    if not text:
        return False
    return any(keyword in text.lower() for keyword in GTA6_KEYWORDS)

def parse_count(s):
    """Parse engagement count strings like '1.2K', '5M', etc."""
    if not s:
        return 0
    s = s.upper().strip()
    try:
        if 'K' in s:
            return int(float(s.replace('K', '')) * 1_000)
        elif 'M' in s:
            return int(float(s.replace('M', '')) * 1_000_000)
        elif 'B' in s:
            return int(float(s.replace('B', '')) * 1_000_000_000)
        else:
            digits = re.sub(r"[^\d.]", "", s)
            return int(float(digits)) if digits and digits != '.' else 0
    except (ValueError, TypeError):
        return 0

async def wait_for_hashtag_page_load(page, timeout=30000):
    """Wait for hashtag page content to load"""
    try:
        # Wait for network to be idle
        await page.wait_for_load_state("networkidle", timeout=timeout)
        
        # Wait for video items to appear
        await page.wait_for_selector(
            'div[data-e2e="recommend-list-item-container"], div[data-e2e*="item"], .video-feed-item',
            timeout=timeout
        )
        
        # Additional wait for dynamic content
        await asyncio.sleep(3)
        return True
    except Exception as e:
        print(f"Hashtag page load timeout: {e}")
        return False

async def get_random_video_from_hashtag_page(page):
    """Get a random video URL from the current hashtag page"""
    try:
        # Multiple selectors to find video items
        video_selectors = [
            'div[data-e2e="recommend-list-item-container"]',
            'div[data-e2e*="item"]',
            'div[class*="video-feed-item"]',
            '[data-e2e="video-feed"] > div',
            '.video-feed-item',
            '[data-e2e*="video"]',
            'div[data-e2e="search_top-item"]'
        ]
        
        video_items = []
        for selector in video_selectors:
            try:
                items = await page.query_selector_all(selector)
                if items:
                    video_items = items
                    print(f"Found {len(items)} video items using selector: {selector}")
                    break
            except Exception:
                continue
        
        if not video_items:
            print("No video items found on hashtag page")
            return None
            
        # Select a random video item
        random_item = random.choice(video_items)
        
        # Extract video URL from the item
        link_selectors = [
            'a[href*="/video/"]', 
            'a[data-e2e="video-title"]',
            'a[data-e2e="video-link"]',
            'a[href*="@"]',
            'a'
        ]
        
        video_url = None
        for selector in link_selectors:
            try:
                link_el = await random_item.query_selector(selector)
                if link_el:
                    url = await link_el.get_attribute("href")
                    if url:
                        # Ensure URL is properly formatted
                        if url.startswith('/'):
                            video_url = f"https://www.tiktok.com{url}"
                        elif not url.startswith('http'):
                            video_url = f"https://www.tiktok.com/{url}"
                        else:
                            video_url = url
                        break
            except Exception:
                continue
                
        return video_url
        
    except Exception as e:
        print(f"Error getting random video from hashtag page: {e}")
        return None

async def navigate_to_hashtag_page(page):
    """Navigate to the hashtag page and wait for it to load"""
    hashtag_url = f"https://www.tiktok.com/tag/{HASHTAG}"
    
    try:
        print(f"Navigating to hashtag page: {hashtag_url}")
        response = await page.goto(hashtag_url, wait_until="domcontentloaded", timeout=30000)
        
        if not response or response.status >= 400:
            print(f"Failed to load hashtag page: {response.status if response else 'No response'}")
            return False
            
        # Wait for page to load
        content_loaded = await wait_for_hashtag_page_load(page)
        if not content_loaded:
            print("Warning: Hashtag page content may not have loaded properly")
            
        # Random pause
        await asyncio.sleep(random.uniform(*RETURN_TO_HASHTAG_PAUSE_RANGE))
        
        # Check if we're blocked or redirected
        current_url = page.url
        if "login" in current_url.lower() or "captcha" in current_url.lower():
            print(f"Warning: May be blocked or redirected to login/captcha")
            return False
            
        return True
        
    except Exception as e:
        print(f"Error navigating to hashtag page: {e}")
        return False

async def extract_comments_from_video(page, video_url, max_comments=20):
    """Extract top-level comments from a specific video page"""
    comments = []
    try:
        print(f"Extracting comments from: {video_url}")
        
        # Navigate to video page
        response = await page.goto(video_url, wait_until="domcontentloaded", timeout=60000)
        
        if not response or response.status >= 400:
            print(f"Failed to load video page: {response.status if response else 'No response'}")
            return comments
            
        # Wait for page to load
        await asyncio.sleep(3)
        
        # Wait for comments section to load
        try:
            await page.wait_for_selector('[data-e2e="comment-level-1"]', timeout=10000)
            print("Comments section loaded successfully")
        except Exception:
            print("Comments section not found or took too long to load")
        
        # Scroll to load more comments
        scroll_attempts = 0
        max_scroll_attempts = 5
        
        while scroll_attempts < max_scroll_attempts and len(comments) < max_comments:
            # Scroll down to load more comments
            await page.mouse.wheel(0, 1500)
            await asyncio.sleep(1.5)
            scroll_attempts += 1
            
            # Extract comments
            comment_selector = '[data-e2e="comment-level-1"]'
            
            try:
                comment_elements = await page.query_selector_all(comment_selector)
                if comment_elements:
                    print(f"Found {len(comment_elements)} comment elements")
                else:
                    print("No comment elements found")
                    comment_elements = []
            except Exception as e:
                print(f"Error finding comment elements: {e}")
                comment_elements = []
            
            # Extract text from comment elements
            new_comments_found = 0
            for el in comment_elements:
                if len(comments) >= max_comments:
                    break
                    
                try:
                    comment_text = ""
                    
                    # Try to get text from spans within the comment
                    text_spans = await el.query_selector_all('span')
                    for span in text_spans:
                        try:
                            span_text = await span.inner_text()
                            if span_text and len(span_text) > len(comment_text):
                                comment_text = span_text
                        except Exception:
                            continue
                    
                    # Fallback: get all text from the comment element
                    if not comment_text:
                        try:
                            comment_text = await el.inner_text()
                        except Exception:
                            try:
                                comment_text = await el.text_content()
                            except Exception:
                                continue
                    
                    # Clean and validate comment text
                    if comment_text:
                        comment_text = comment_text.strip()
                        lines = [line.strip() for line in comment_text.split('\n') if line.strip()]
                        
                        # Look for actual comment content
                        actual_comment = ""
                        for line in lines:
                            if (len(line) > 10 and 
                                not re.match(r'^\d+[KMkm]?$', line) and 
                                not re.match(r'^@\w+$', line) and
                                not re.match(r'^\d+[hdwmy]$', line)):
                                actual_comment = line
                                break
                        
                        # Add comment if it's valid and not a duplicate
                        if actual_comment and actual_comment not in [c['text'] for c in comments]:
                            comments.append({
                                'text': actual_comment,
                                'extracted_at': datetime.now().isoformat()
                            })
                            new_comments_found += 1
                            print(f"Extracted comment: {actual_comment[:60]}...")
                            
                except Exception as e:
                    print(f"Error extracting individual comment: {e}")
                    continue
            
            if new_comments_found == 0:
                print(f"No new comments found in scroll attempt {scroll_attempts}")
            else:
                print(f"Found {new_comments_found} new comments in this scroll")
                
    except Exception as e:
        print(f"Error extracting comments from {video_url}: {e}")
    
    print(f"Total comments extracted: {len(comments)}")
    return comments

async def extract_engagement_metrics_from_video_page(page):
    """Extract engagement metrics from the current video page"""
    
    # Selectors for video page (different from feed page)
    selectors = {
        'likes': [
            '[data-e2e="like-count"]',
            'strong[data-e2e="like-count"]',
            '[data-e2e="browse-like-count"]',
            'strong[data-e2e="browse-like-count"]',
            '[title*="like" i]',
            '[aria-label*="like" i]',
            'span[data-e2e*="like"]',
            'strong[data-e2e*="like"]',
            'svg[fill="#ff0050"] + strong',
            'svg[fill="#FE2C55"] + strong',
        ],
        'comments': [
            '[data-e2e="comment-count"]',
            'strong[data-e2e="comment-count"]',
            '[data-e2e="browse-comment-count"]',
            'strong[data-e2e="browse-comment-count"]',
            '[title*="comment" i]',
            '[aria-label*="comment" i]',
            'span[data-e2e*="comment"]',
            'strong[data-e2e*="comment"]',
        ],
        'saves': [
            '[data-e2e="undefined-count"]',
            'strong[data-e2e="undefined-count"]',
            '[data-e2e="share-count"]',
            '[data-e2e="save-count"]',
            '[title*="share" i]',
            '[title*="save" i]',
            'span[data-e2e*="share"]',
            'span[data-e2e*="save"]',
            'strong[data-e2e*="share"]',
            'strong[data-e2e*="save"]',
        ]
    }

    async def find_metric_text(selector_list, metric_name):
        """Find element and extract text"""
        for selector in selector_list:
            try:
                element = await page.query_selector(selector)
                if element:
                    # Try multiple methods to extract text
                    text_methods = [
                        element.inner_text,
                        element.text_content,
                        lambda: element.get_attribute("title"),
                        lambda: element.get_attribute("aria-label")
                    ]
                    
                    for method in text_methods:
                        try:
                            text = await method()
                            if text and text.strip():
                                cleaned_text = re.sub(r'[^\d.KMBkmb]', '', text.strip())
                                if cleaned_text:
                                    return cleaned_text
                        except Exception:
                            continue
                        
            except Exception:
                continue
        return ""

    # Extract all metrics
    likes_text = await find_metric_text(selectors['likes'], 'likes')
    comments_text = await find_metric_text(selectors['comments'], 'comments')
    saves_text = await find_metric_text(selectors['saves'], 'saves')
    
    # Fallback: try to find any strong elements with numbers
    if not likes_text or not comments_text or not saves_text:
        try:
            strong_elements = await page.query_selector_all('strong')
            engagement_texts = []
            
            for strong in strong_elements:
                try:
                    text = await strong.inner_text()
                    if text and re.match(r'^\d+[KMB]?$', text.strip()):
                        engagement_texts.append(text.strip())
                except Exception:
                    continue
            
            # Assign based on position (usually likes, comments, saves)
            if len(engagement_texts) >= 1 and not likes_text:
                likes_text = engagement_texts[0]
            if len(engagement_texts) >= 2 and not comments_text:
                comments_text = engagement_texts[1]
            if len(engagement_texts) >= 3 and not saves_text:
                saves_text = engagement_texts[2]
                
        except Exception as e:
            print(f"Error in fallback engagement extraction: {e}")

    return {
        'likes': parse_count(likes_text),
        'comments': parse_count(comments_text),
        'saves': parse_count(saves_text),
        'raw_likes': likes_text,
        'raw_comments': comments_text,
        'raw_saves': saves_text
    }

async def extract_video_caption_from_video_page(page):
    """Extract video caption from the current video page"""
    caption_selectors = [
        '[data-e2e="browse-video-desc"]',
        '[data-e2e="video-desc"]',
        '.video-meta-caption',
        '.tt-video-meta-caption',
        '[data-e2e="browse-video-desc"] span',
        'h1[data-e2e="browse-video-desc"]'
    ]
    
    for selector in caption_selectors:
        try:
            caption_el = await page.query_selector(selector)
            if caption_el:
                try:
                    caption = await caption_el.inner_text()
                    if caption and caption.strip():
                        return caption.strip()
                except Exception:
                    try:
                        caption = await caption_el.text_content()
                        if caption and caption.strip():
                            return caption.strip()
                    except Exception:
                        continue
        except Exception:
            continue
    
    return ""

async def scrape_single_video(page, video_url):
    """Scrape a single video and return its data"""
    try:
        print(f"Scraping video: {video_url}")
        
        # Navigate to video page
        response = await page.goto(video_url, wait_until="domcontentloaded", timeout=30000)
        
        if not response or response.status >= 400:
            print(f"Failed to load video page: {response.status if response else 'No response'}")
            return None
            
        # Wait for page to load
        await asyncio.sleep(3)
        
        # Extract video ID from URL
        vid_id_match = re.search(r'/video/(\d+)', video_url)
        if not vid_id_match:
            vid_id_match = re.search(r'@[\w.-]+/video/(\d+)', video_url)
            if not vid_id_match:
                print(f"Could not extract video ID from URL: {video_url}")
                return None
                
        video_id = vid_id_match.group(1)
        
        # Extract caption and check if it's GTA6 related
        caption = await extract_video_caption_from_video_page(page)
        if not contains_gta6(caption):
            print(f"Video {video_id} is not GTA6 related, skipping")
            return None
        
        # Extract engagement metrics
        engagement_data = await extract_engagement_metrics_from_video_page(page)
        
        # Extract comments if enabled
        comment_data = []
        if COMMENT_EXTRACTION_ENABLED:
            try:
                comment_data = await extract_comments_from_video(page, video_url, MAX_COMMENTS_PER_VIDEO)
            except Exception as e:
                print(f"Error extracting comments for video {video_id}: {e}")

        # Format comment data for CSV
        comments_text_field = ""
        comments_json_field = ""
        
        if comment_data:
            comments_text_field = " ||| ".join([c['text'] for c in comment_data])
            comments_json_field = str([{
                'text': c['text'][:500],
                'extracted_at': c['extracted_at']
            } for c in comment_data])

        print(f"Video {video_id} engagement - Likes: {engagement_data['likes']} ({engagement_data['raw_likes']}), "
              f"Comments: {engagement_data['comments']} ({engagement_data['raw_comments']}), "
              f"Saves: {engagement_data['saves']} ({engagement_data['raw_saves']})")

        return {
            "video_id": video_id,
            "video_url": video_url,
            "caption": caption,
            "likes": engagement_data['likes'],
            "comments": engagement_data['comments'],
            "saves": engagement_data['saves'],
            "raw_likes": engagement_data['raw_likes'],
            "raw_comments": engagement_data['raw_comments'],
            "raw_saves": engagement_data['raw_saves'],
            "comment_text": comments_text_field,
            "comments_data": comments_json_field,
            "comment_count_extracted": len(comment_data),
            "timestamp": datetime.now().isoformat(),
        }
        
    except Exception as e:
        print(f"Error scraping video {video_url}: {e}")
        return None

def upload_to_gcs(filepath):
    """Upload CSV file to Google Cloud Storage"""
    try:
        if not os.path.exists(GCP_CREDENTIALS_PATH):
            print(f"Warning: GCP credentials file not found at {GCP_CREDENTIALS_PATH}")
            print("Skipping upload to GCS")
            return
            
        client = storage.Client.from_service_account_json(GCP_CREDENTIALS_PATH)
        bucket = client.bucket(GCS_BUCKET_NAME)
        blob = bucket.blob(GCS_DEST_BLOB)
        blob.upload_from_filename(filepath)
        print(f"Successfully uploaded to gs://{GCS_BUCKET_NAME}/{GCS_DEST_BLOB}")
        
    except Exception as e:
        print(f"Error uploading to GCS: {e}")

async def main():
    """Main function to run the scraper"""
    # Create data directory
    os.makedirs(DATA_DIR, exist_ok=True)
    csv_path = os.path.join(DATA_DIR, BASE_FILENAME)
    
    print(f"Starting cyclic scraper for #{HASHTAG}")
    print(f"Target videos: {MAX_VIDEOS_TOTAL}")
    print(f"Data will be saved to: {csv_path}")
    print(f"Comment extraction: {'ENABLED' if COMMENT_EXTRACTION_ENABLED else 'DISABLED'}")
    if COMMENT_EXTRACTION_ENABLED:
        print(f"Max comments per video: {MAX_COMMENTS_PER_VIDEO}")
    
    # Load existing video IDs to avoid duplicates
    existing_video_ids = load_existing_video_ids(DATA_DIR, BASE_FILENAME)
    
    async with async_playwright() as p:
        # Launch browser
        browser = await p.chromium.launch(
            headless=True,  # Set to False for debugging
            args=[
                '--no-sandbox',
                '--disable-blink-features=AutomationControlled',
                '--disable-web-security',
                '--disable-features=VizDisplayCompositor',
                '--disable-dev-shm-usage',
                '--user-agent=' + random.choice(USER_AGENTS)
            ]
        )
        
        # Create context
        context = await browser.new_context(
            user_agent=random.choice(USER_AGENTS),
            viewport={'width': 1920, 'height': 1080},
            locale='en-US',
            timezone_id='America/New_York'
        )
        
        page = await context.new_page()
        
        # Set additional headers
        await page.set_extra_http_headers({
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Cache-Control': 'no-cache',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none'
        })
        
        session_data = []  # Data collected in this session
        scraped_video_ids = existing_video_ids.copy()  # Include existing IDs
        failed_attempts = 0
        max_failed_attempts = 50
        videos_scraped_this_session = 0  # Track videos scraped in current session
        
        try:
            while videos_scraped_this_session < MAX_VIDEOS_TOTAL and failed_attempts < max_failed_attempts:
                print(f"\n{'='*60}")
                print(f"Cycle {videos_scraped_this_session + 1} - Returning to #{HASHTAG} page")
                print(f"Session collected: {videos_scraped_this_session}/{MAX_VIDEOS_TOTAL} videos")
                print(f"Total videos in dataset: {len(scraped_video_ids)}")
                print(f"{'='*60}")
                
                # Navigate to hashtag page
                if not await navigate_to_hashtag_page(page):
                    print("Failed to navigate to hashtag page")
                    failed_attempts += 1
                    continue
                
                # Get a random video URL from the hashtag page
                video_url = await get_random_video_from_hashtag_page(page)
                if not video_url:
                    print("Could not find a video URL on hashtag page")
                    failed_attempts += 1
                    continue
                
                # Extract video ID to check for duplicates
                vid_id_match = re.search(r'/video/(\d+)', video_url)
                if vid_id_match:
                    video_id = vid_id_match.group(1)
                    if video_id in scraped_video_ids:
                        print(f"Video {video_id} already scraped, skipping")
                        # This is not a failure, just continue to next video
                        continue
                
                # Scrape the video
                video_data = await scrape_single_video(page, video_url)
                
                if video_data:
                    session_data.append(video_data)
                    scraped_video_ids.add(video_data['video_id'])
                    videos_scraped_this_session += 1
                    comments_info = f" ({video_data['comment_count_extracted']} comments)" if COMMENT_EXTRACTION_ENABLED else ""
                    print(f"✅ Successfully scraped video {video_data['video_id']}{comments_info}")
                    failed_attempts = 0  # Reset failed attempts on success
                    
                    # Save data after every 10 videos or at end
                    if len(session_data) % 10 == 0 or videos_scraped_this_session >= MAX_VIDEOS_TOTAL:
                        print(f"Saving {len(session_data)} videos to CSV...")
                        saved_path = save_data_with_append(session_data, DATA_DIR, BASE_FILENAME)
                        if saved_path:
                            csv_path = saved_path
                            # Upload to GCS periodically
                            upload_to_gcs(csv_path)
                        session_data = []  # Clear session data after saving
                else:
                    print("❌ Failed to scrape video or video not GTA6 related")
                    failed_attempts += 1
                
                # Random delay before next cycle
                delay = random.uniform(3, 8)
                print(f"Waiting {delay:.1f} seconds before next cycle...")
                await asyncio.sleep(delay)
                
        except KeyboardInterrupt:
            print("\nScraping interrupted by user")
        except Exception as e:
            print(f"Error during scraping: {e}")
        finally:
            await browser.close()
        
        # Save any remaining data
        if session_data:
            print(f"Saving final {len(session_data)} videos to CSV...")
            saved_path = save_data_with_append(session_data, DATA_DIR, BASE_FILENAME)
            if saved_path:
                csv_path = saved_path
                upload_to_gcs(csv_path)
        
        # Final statistics
        print(f"\n{'='*60}")
        print(f"SCRAPING SESSION COMPLETED")
        print(f"{'='*60}")
        
        # Count total videos in the dataset
        total_videos = len(load_existing_video_ids(DATA_DIR, BASE_FILENAME))
        print(f"Videos scraped this session: {videos_scraped_this_session}")
        print(f"Total videos in dataset: {total_videos}")
        print(f"Dataset location: {csv_path}")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nScraping interrupted by user")
    except Exception as e:
        print(f"Fatal error: {e}")
        raise