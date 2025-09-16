"""
Image Processing Pipeline for crypto signal extraction from images.
Includes OCR integration and chart analysis capabilities.
"""

import asyncio
import io
import os
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import hashlib
from datetime import datetime
import json

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import pytesseract
import easyocr
from google.cloud import vision
import structlog
from tenacity import retry, stop_after_attempt, wait_exponential
import aiohttp
import matplotlib.pyplot as plt
from scipy import signal
from sklearn.cluster import DBSCAN

from src.config.settings import settings
from src.core.redis_client import signal_cache
from src.core.database import get_db_context

logger = structlog.get_logger()

# Initialize OCR readers
EASYOCR_READER = None
GOOGLE_VISION_CLIENT = None

# Check for Google Cloud Vision credentials
HAS_GOOGLE_VISION = bool(os.environ.get('GOOGLE_CLOUD_VISION_API_KEY'))


class ImageQuality:
    """Image quality assessment utilities."""
    
    @staticmethod
    def assess_quality(image: np.ndarray) -> Dict[str, Any]:
        """Assess image quality metrics."""
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Calculate metrics
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        mean_brightness = np.mean(gray)
        contrast = gray.std()
        
        # Detect blur
        is_blurry = laplacian_var < 100
        
        # Detect low contrast
        is_low_contrast = contrast < 30
        
        # Detect too dark/bright
        is_too_dark = mean_brightness < 50
        is_too_bright = mean_brightness > 200
        
        return {
            'laplacian_variance': laplacian_var,
            'mean_brightness': mean_brightness,
            'contrast': contrast,
            'is_blurry': is_blurry,
            'is_low_contrast': is_low_contrast,
            'is_too_dark': is_too_dark,
            'is_too_bright': is_too_bright,
            'quality_score': min(100, (laplacian_var / 10) * (contrast / 100) * 100)
        }
    
    @staticmethod
    def enhance_image(image: Image.Image, quality_metrics: Dict[str, Any]) -> Image.Image:
        """Enhance image based on quality metrics."""
        enhanced = image
        
        # Adjust brightness if needed
        if quality_metrics['is_too_dark']:
            enhancer = ImageEnhance.Brightness(enhanced)
            enhanced = enhancer.enhance(1.5)
        elif quality_metrics['is_too_bright']:
            enhancer = ImageEnhance.Brightness(enhanced)
            enhanced = enhancer.enhance(0.8)
        
        # Adjust contrast if needed
        if quality_metrics['is_low_contrast']:
            enhancer = ImageEnhance.Contrast(enhanced)
            enhanced = enhancer.enhance(1.5)
        
        # Apply sharpening if blurry
        if quality_metrics['is_blurry']:
            enhanced = enhanced.filter(ImageFilter.SHARPEN)
        
        return enhanced


class OCRProcessor:
    """Multi-engine OCR processor with fallback support."""
    
    def __init__(self):
        self.google_vision_client = None
        self.easyocr_reader = None
        self.tesseract_available = self._check_tesseract()
        
        # Initialize Google Vision if credentials available
        if HAS_GOOGLE_VISION:
            try:
                self.google_vision_client = vision.ImageAnnotatorClient()
                logger.info("Google Vision API initialized")
            except Exception as e:
                logger.error(f"Failed to initialize Google Vision: {e}")
        
        # Initialize EasyOCR as fallback
        try:
            self.easyocr_reader = easyocr.Reader(['en', 'ru'])
            logger.info("EasyOCR initialized")
        except Exception as e:
            logger.error(f"Failed to initialize EasyOCR: {e}")
    
    def _check_tesseract(self) -> bool:
        """Check if Tesseract is available."""
        try:
            pytesseract.get_tesseract_version()
            return True
        except Exception:
            return False
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def extract_text(self, image: Image.Image, confidence_threshold: float = 0.5) -> Dict[str, Any]:
        """Extract text from image using multiple OCR engines."""
        results = {
            'text': '',
            'confidence': 0.0,
            'engine': 'none',
            'languages_detected': [],
            'text_blocks': []
        }
        
        # Try Google Vision first (best accuracy)
        if self.google_vision_client:
            try:
                result = await self._google_vision_ocr(image)
                if result['confidence'] > confidence_threshold:
                    return result
            except Exception as e:
                logger.warning(f"Google Vision OCR failed: {e}")
        
        # Try EasyOCR as second option
        if self.easyocr_reader:
            try:
                result = await self._easyocr_ocr(image)
                if result['confidence'] > confidence_threshold:
                    return result
            except Exception as e:
                logger.warning(f"EasyOCR failed: {e}")
        
        # Fallback to Tesseract
        if self.tesseract_available:
            try:
                result = await self._tesseract_ocr(image)
                return result
            except Exception as e:
                logger.warning(f"Tesseract OCR failed: {e}")
        
        return results
    
    async def _google_vision_ocr(self, image: Image.Image) -> Dict[str, Any]:
        """Use Google Vision API for OCR."""
        # Convert PIL image to bytes
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()
        
        # Create Vision API image
        vision_image = vision.Image(content=img_byte_arr)
        
        # Perform text detection
        response = self.google_vision_client.text_detection(image=vision_image)
        texts = response.text_annotations
        
        if texts:
            full_text = texts[0].description
            
            # Calculate confidence (Google Vision doesn't provide direct confidence)
            confidence = 0.85  # Default high confidence for Google Vision
            
            # Extract text blocks
            text_blocks = []
            for text in texts[1:]:
                vertices = [(v.x, v.y) for v in text.bounding_poly.vertices]
                text_blocks.append({
                    'text': text.description,
                    'bbox': vertices,
                    'confidence': confidence
                })
            
            # Detect language
            response = self.google_vision_client.document_text_detection(image=vision_image)
            languages = []
            if response.full_text_annotation:
                for page in response.full_text_annotation.pages:
                    for block in page.blocks:
                        if block.property and block.property.detected_languages:
                            for lang in block.property.detected_languages:
                                if lang.language_code not in languages:
                                    languages.append(lang.language_code)
            
            return {
                'text': full_text,
                'confidence': confidence,
                'engine': 'google_vision',
                'languages_detected': languages,
                'text_blocks': text_blocks
            }
        
        return {
            'text': '',
            'confidence': 0.0,
            'engine': 'google_vision',
            'languages_detected': [],
            'text_blocks': []
        }
    
    async def _easyocr_ocr(self, image: Image.Image) -> Dict[str, Any]:
        """Use EasyOCR for text extraction."""
        # Convert PIL image to numpy array
        img_array = np.array(image)
        
        # Run OCR
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, 
            self.easyocr_reader.readtext, 
            img_array
        )
        
        if result:
            # Extract text and calculate average confidence
            texts = []
            confidences = []
            text_blocks = []
            
            for (bbox, text, conf) in result:
                texts.append(text)
                confidences.append(conf)
                text_blocks.append({
                    'text': text,
                    'bbox': bbox,
                    'confidence': conf
                })
            
            full_text = ' '.join(texts)
            avg_confidence = np.mean(confidences) if confidences else 0.0
            
            return {
                'text': full_text,
                'confidence': avg_confidence,
                'engine': 'easyocr',
                'languages_detected': ['en'],  # EasyOCR was initialized with English
                'text_blocks': text_blocks
            }
        
        return {
            'text': '',
            'confidence': 0.0,
            'engine': 'easyocr',
            'languages_detected': [],
            'text_blocks': []
        }
    
    async def _tesseract_ocr(self, image: Image.Image) -> Dict[str, Any]:
        """Use Tesseract for text extraction."""
        # Run OCR with confidence scores
        loop = asyncio.get_event_loop()
        data = await loop.run_in_executor(
            None,
            pytesseract.image_to_data,
            image,
            output_type=pytesseract.Output.DICT
        )
        
        # Extract text and confidence
        texts = []
        confidences = []
        text_blocks = []
        
        for i in range(len(data['text'])):
            if int(data['conf'][i]) > 0:
                texts.append(data['text'][i])
                confidences.append(int(data['conf'][i]) / 100.0)
                
                text_blocks.append({
                    'text': data['text'][i],
                    'bbox': [
                        (data['left'][i], data['top'][i]),
                        (data['left'][i] + data['width'][i], data['top'][i] + data['height'][i])
                    ],
                    'confidence': int(data['conf'][i]) / 100.0
                })
        
        full_text = ' '.join(texts)
        avg_confidence = np.mean(confidences) if confidences else 0.0
        
        # Detect language
        try:
            lang = pytesseract.image_to_osd(image)
            language = lang.split('Script:')[1].split('\n')[0].strip().lower()
        except:
            language = 'en'
        
        return {
            'text': full_text,
            'confidence': avg_confidence,
            'engine': 'tesseract',
            'languages_detected': [language],
            'text_blocks': text_blocks
        }


class ChartAnalyzer:
    """Analyzes trading charts and extracts patterns."""
    
    def __init__(self):
        self.chart_patterns = {
            'candlestick': self._detect_candlestick,
            'line': self._detect_line_chart,
            'bar': self._detect_bar_chart
        }
    
    async def analyze_chart(self, image: np.ndarray) -> Dict[str, Any]:
        """Analyze trading chart and extract information."""
        results = {
            'chart_type': None,
            'price_levels': [],
            'support_resistance': [],
            'trend': None,
            'patterns': [],
            'indicators': [],
            'time_frame': None
        }
        
        # Detect chart type
        chart_type = await self._detect_chart_type(image)
        results['chart_type'] = chart_type
        
        if chart_type:
            # Extract price levels
            results['price_levels'] = await self._extract_price_levels(image)
            
            # Detect support and resistance
            results['support_resistance'] = await self._detect_support_resistance(image)
            
            # Analyze trend
            results['trend'] = await self._analyze_trend(image)
            
            # Detect patterns
            results['patterns'] = await self._detect_patterns(image, chart_type)
            
            # Detect technical indicators
            results['indicators'] = await self._detect_indicators(image)
        
        return results
    
    async def _detect_chart_type(self, image: np.ndarray) -> Optional[str]:
        """Detect the type of chart in the image."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Use edge detection to find chart elements
        edges = cv2.Canny(gray, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Analyze contour patterns
        rect_count = 0
        line_count = 0
        
        for contour in contours:
            # Approximate contour
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            if len(approx) == 4:
                rect_count += 1
            elif len(approx) == 2:
                line_count += 1
        
        # Determine chart type based on patterns
        if rect_count > 10:
            return 'candlestick'
        elif line_count > rect_count:
            return 'line'
        elif rect_count > 5:
            return 'bar'
        
        return None
    
    async def _extract_price_levels(self, image: np.ndarray) -> List[float]:
        """Extract price levels from chart."""
        price_levels = []
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Detect horizontal lines (price levels)
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10)
        
        if lines is not None:
            # Group horizontal lines
            horizontal_lines = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if abs(y2 - y1) < 5:  # Nearly horizontal
                    horizontal_lines.append(y1)
            
            # Cluster similar y-coordinates
            if horizontal_lines:
                horizontal_lines = np.array(horizontal_lines).reshape(-1, 1)
                clustering = DBSCAN(eps=5, min_samples=2).fit(horizontal_lines)
                
                # Get unique clusters
                unique_labels = set(clustering.labels_)
                for label in unique_labels:
                    if label != -1:  # Ignore noise
                        cluster_points = horizontal_lines[clustering.labels_ == label]
                        price_levels.append(float(np.mean(cluster_points)))
        
        return sorted(price_levels)
    
    async def _detect_support_resistance(self, image: np.ndarray) -> Dict[str, List[float]]:
        """Detect support and resistance levels."""
        price_levels = await self._extract_price_levels(image)
        
        # Simple heuristic: top levels are resistance, bottom are support
        if len(price_levels) >= 2:
            mid_point = len(price_levels) // 2
            return {
                'support': price_levels[:mid_point],
                'resistance': price_levels[mid_point:]
            }
        
        return {'support': [], 'resistance': []}
    
    async def _analyze_trend(self, image: np.ndarray) -> str:
        """Analyze overall trend direction."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Get vertical profile (sum along horizontal axis)
        profile = np.sum(gray, axis=1)
        
        # Smooth the profile
        from scipy.ndimage import gaussian_filter1d
        smoothed = gaussian_filter1d(profile, sigma=5)
        
        # Calculate trend using linear regression
        x = np.arange(len(smoothed))
        z = np.polyfit(x, smoothed, 1)
        slope = z[0]
        
        # Determine trend based on slope
        if slope > 0.1:
            return 'uptrend'
        elif slope < -0.1:
            return 'downtrend'
        else:
            return 'sideways'
    
    async def _detect_patterns(self, image: np.ndarray, chart_type: str) -> List[str]:
        """Detect chart patterns."""
        patterns = []
        
        if chart_type == 'candlestick':
            patterns.extend(await self._detect_candlestick_patterns(image))
        elif chart_type == 'line':
            patterns.extend(await self._detect_line_patterns(image))
        
        return patterns
    
    async def _detect_candlestick_patterns(self, image: np.ndarray) -> List[str]:
        """Detect candlestick patterns."""
        patterns = []
        
        # Placeholder for pattern detection logic
        # In production, this would use more sophisticated pattern recognition
        
        # Example patterns to detect:
        # - Doji
        # - Hammer
        # - Shooting Star
        # - Engulfing patterns
        
        return patterns
    
    async def _detect_line_patterns(self, image: np.ndarray) -> List[str]:
        """Detect line chart patterns."""
        patterns = []
        
        # Placeholder for pattern detection
        # Would detect: Head and Shoulders, Triangles, Flags, etc.
        
        return patterns
    
    async def _detect_indicators(self, image: np.ndarray) -> List[str]:
        """Detect technical indicators present in the chart."""
        indicators = []
        
        # Look for common indicator patterns
        # This is a simplified version - real implementation would be more sophisticated
        
        # Convert to HSV for color-based detection
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Check for multiple colored lines (could be MA, EMA, etc.)
        unique_colors = len(np.unique(hsv[:, :, 0]))
        
        if unique_colors > 10:
            indicators.append('moving_averages')
        
        # Check for volume bars at bottom
        bottom_section = image[int(image.shape[0] * 0.8):, :]
        if self._has_bars(bottom_section):
            indicators.append('volume')
        
        return indicators
    
    def _detect_candlestick(self, image: np.ndarray) -> bool:
        """Detect if image contains candlestick patterns."""
        # Implementation placeholder
        return False
    
    def _detect_line_chart(self, image: np.ndarray) -> bool:
        """Detect if image contains line chart."""
        # Implementation placeholder
        return False
    
    def _detect_bar_chart(self, image: np.ndarray) -> bool:
        """Detect if image contains bar chart."""
        # Implementation placeholder
        return False
    
    def _has_bars(self, image: np.ndarray) -> bool:
        """Check if image section contains bar patterns."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        edges = cv2.Canny(gray, 50, 150)
        
        # Count vertical lines
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=20, maxLineGap=5)
        
        if lines is not None:
            vertical_lines = 0
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if abs(x2 - x1) < 5:  # Nearly vertical
                    vertical_lines += 1
            
            return vertical_lines > 5
        
        return False


class ImageProcessor:
    """Main image processing coordinator."""
    
    def __init__(self):
        self.ocr_processor = OCRProcessor()
        self.chart_analyzer = ChartAnalyzer()
        self.cache_dir = Path("data/processed_images")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    async def process_image(self, image_path: str, message_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process an image and extract all relevant information."""
        try:
            # Load image
            image = Image.open(image_path)
            img_array = np.array(image)
            
            # Assess quality
            quality_metrics = ImageQuality.assess_quality(img_array)
            
            # Enhance if needed
            if quality_metrics['quality_score'] < 70:
                image = ImageQuality.enhance_image(image, quality_metrics)
                img_array = np.array(image)
            
            # Extract text
            ocr_results = await self.ocr_processor.extract_text(image)
            
            # Analyze chart if present
            chart_analysis = await self.chart_analyzer.analyze_chart(img_array)
            
            # Combine results
            results = {
                'image_path': image_path,
                'message_id': message_data.get('message_id'),
                'channel_id': message_data.get('channel_id'),
                'timestamp': datetime.utcnow().isoformat(),
                'quality_metrics': quality_metrics,
                'ocr_results': ocr_results,
                'chart_analysis': chart_analysis,
                'extracted_signals': await self._extract_signals_from_text(ocr_results['text'])
            }
            
            # Cache results
            await self._cache_results(results)
            
            return results
            
        except Exception as e:
            logger.error(f"Error processing image {image_path}: {e}")
            return {
                'error': str(e),
                'image_path': image_path
            }
    
    async def _extract_signals_from_text(self, text: str) -> List[Dict[str, Any]]:
        """Extract trading signals from OCR text."""
        signals = []
        
        if not text:
            return signals
        
        # Pattern matching for common signal formats
        patterns = {
            'entry': r'(?:entry|buy|long|short)[\s:]+([0-9.,]+)',
            'stop_loss': r'(?:sl|stop[\s-]?loss|stop)[\s:]+([0-9.,]+)',
            'take_profit': r'(?:tp|take[\s-]?profit|target)[\s:]+([0-9.,]+)',
            'symbol': r'([A-Z]{3,10}/?[A-Z]{3,4})',
        }
        
        import re
        
        # Extract signal components
        signal_data = {}
        
        for key, pattern in patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                if key in ['entry', 'stop_loss', 'take_profit']:
                    # Clean and convert to float
                    try:
                        signal_data[key] = float(matches[0].replace(',', ''))
                    except:
                        signal_data[key] = matches[0]
                else:
                    signal_data[key] = matches[0]
        
        if signal_data:
            signals.append(signal_data)
        
        return signals
    
    async def _cache_results(self, results: Dict[str, Any]):
        """Cache processing results."""
        # Generate cache key
        cache_key = f"image_processed:{results.get('channel_id')}:{results.get('message_id')}"
        
        # Store in Redis
        await signal_cache.set_signal(cache_key, results, ttl=86400)  # 24 hours
        
        # Also save to file for debugging
        timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        cache_file = self.cache_dir / f"{timestamp}_{results.get('message_id')}.json"
        
        with open(cache_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
    
    async def process_batch(self, image_paths: List[Tuple[str, Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """Process multiple images in parallel."""
        tasks = []
        for image_path, message_data in image_paths:
            tasks.append(self.process_image(image_path, message_data))
        
        results = await asyncio.gather(*tasks)
        return results


# Global processor instance
image_processor = ImageProcessor()


async def process_telegram_image(image_path: str, message_data: Dict[str, Any]) -> Dict[str, Any]:
    """Process a Telegram image."""
    return await image_processor.process_image(image_path, message_data)


async def process_image_batch(images: List[Tuple[str, Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """Process multiple images."""
    return await image_processor.process_batch(images)
