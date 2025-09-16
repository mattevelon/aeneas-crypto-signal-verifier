"""Response Processor for parsing and validating LLM responses."""

import json
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass
import logging
import re

logger = logging.getLogger(__name__)


@dataclass
class ProcessedResponse:
    """Processed and validated LLM response."""
    confidence_score: float
    risk_level: str
    signal_validity: Dict[str, Any]
    optimizations: Dict[str, Any]
    risk_factors: List[str]
    recommendations: List[str]
    justification: Dict[str, str]
    raw_response: str
    processing_status: str
    validation_errors: List[str]
    metadata: Dict[str, Any]


class ResponseProcessor:
    """
    Processes and validates LLM responses with quality scoring.
    Implements response caching and post-processing pipeline.
    """
    
    def __init__(self):
        """Initialize response processor."""
        self.validation_rules = self._initialize_validation_rules()
        self.quality_thresholds = {
            'minimum_confidence': 0,
            'maximum_confidence': 100,
            'valid_risk_levels': ['low', 'medium', 'high'],
            'minimum_justification_length': {
                'novice': 50,
                'intermediate': 200,
                'expert': 500
            }
        }
        
        # Response cache
        self.response_cache = {}
        self.cache_ttl = 3600  # 1 hour
        
        # Quality metrics
        self.quality_scores = []
        
        logger.info("Initialized ResponseProcessor")
    
    def _initialize_validation_rules(self) -> Dict[str, Any]:
        """Initialize validation rules for response fields."""
        return {
            'required_fields': [
                'confidence_score',
                'risk_level',
                'signal_validity',
                'justification'
            ],
            'optional_fields': [
                'optimizations',
                'risk_factors',
                'recommendations'
            ],
            'field_types': {
                'confidence_score': (int, float),
                'risk_level': str,
                'signal_validity': dict,
                'optimizations': dict,
                'risk_factors': list,
                'recommendations': list,
                'justification': dict
            }
        }
    
    def process_response(self, 
                         raw_response: str,
                         response_type: str = 'signal_analysis') -> ProcessedResponse:
        """
        Process and validate LLM response.
        
        Args:
            raw_response: Raw LLM response string
            response_type: Type of response to process
            
        Returns:
            Processed and validated response
        """
        # Try to parse JSON response
        parsed_data, parse_errors = self._parse_json_response(raw_response)
        
        if parse_errors:
            # Try to extract data from non-JSON response
            parsed_data = self._extract_from_text(raw_response)
        
        # Validate response structure
        validation_errors = self._validate_response(parsed_data, response_type)
        
        # Extract and process fields
        processed = self._extract_fields(parsed_data, raw_response)
        
        # Score response quality
        quality_score = self._score_quality(processed)
        processed.metadata['quality_score'] = quality_score
        
        # Determine processing status
        if validation_errors:
            processed.processing_status = 'partial'
            processed.validation_errors = validation_errors
        else:
            processed.processing_status = 'complete'
        
        # Cache processed response
        self._cache_response(raw_response, processed)
        
        # Track quality metrics
        self.quality_scores.append(quality_score)
        if len(self.quality_scores) > 100:
            self.quality_scores.pop(0)
        
        return processed
    
    def _parse_json_response(self, raw_response: str) -> Tuple[Dict[str, Any], List[str]]:
        """Parse JSON from LLM response."""
        errors = []
        
        # Clean response (remove markdown code blocks if present)
        cleaned = raw_response.strip()
        if cleaned.startswith('```json'):
            cleaned = cleaned[7:]
        if cleaned.startswith('```'):
            cleaned = cleaned[3:]
        if cleaned.endswith('```'):
            cleaned = cleaned[:-3]
        
        # Try to parse JSON
        try:
            data = json.loads(cleaned)
            return data, []
        except json.JSONDecodeError as e:
            errors.append(f"JSON parse error: {e}")
            
            # Try to find JSON object in response
            json_match = re.search(r'\{.*\}', cleaned, re.DOTALL)
            if json_match:
                try:
                    data = json.loads(json_match.group())
                    return data, []
                except json.JSONDecodeError:
                    errors.append("Failed to extract valid JSON from response")
        
        return {}, errors
    
    def _extract_from_text(self, text: str) -> Dict[str, Any]:
        """Extract structured data from non-JSON text response."""
        extracted = {}
        
        # Extract confidence score
        confidence_match = re.search(r'confidence[:\s]+(\d+(?:\.\d+)?)', text, re.IGNORECASE)
        if confidence_match:
            try:
                extracted['confidence_score'] = float(confidence_match.group(1))
            except ValueError:
                pass
        
        # Extract risk level
        risk_match = re.search(r'risk[:\s]+(low|medium|high)', text, re.IGNORECASE)
        if risk_match:
            extracted['risk_level'] = risk_match.group(1).lower()
        
        # Extract validity
        valid_match = re.search(r'(valid|invalid|not valid)', text, re.IGNORECASE)
        if valid_match:
            extracted['signal_validity'] = {
                'is_valid': 'valid' in valid_match.group(1).lower(),
                'reasons': []
            }
        
        # Extract recommendations (bullet points)
        recommendations = re.findall(r'[-•*]\s*([^-•*\n]+)', text)
        if recommendations:
            extracted['recommendations'] = [r.strip() for r in recommendations[:5]]
        
        # Extract risk factors
        risk_factors = []
        if 'high volatility' in text.lower():
            risk_factors.append('high_volatility')
        if 'low liquidity' in text.lower():
            risk_factors.append('low_liquidity')
        if 'manipulation' in text.lower():
            risk_factors.append('manipulation_risk')
        if risk_factors:
            extracted['risk_factors'] = risk_factors
        
        # Basic justification
        extracted['justification'] = {
            'novice': text[:200] if len(text) > 200 else text,
            'intermediate': text[:500] if len(text) > 500 else text,
            'expert': text
        }
        
        return extracted
    
    def _validate_response(self, data: Dict[str, Any], response_type: str) -> List[str]:
        """Validate response structure and content."""
        errors = []
        
        # Check required fields
        for field in self.validation_rules['required_fields']:
            if field not in data:
                errors.append(f"Missing required field: {field}")
        
        # Validate field types
        for field, expected_type in self.validation_rules['field_types'].items():
            if field in data:
                if not isinstance(data[field], expected_type):
                    errors.append(f"Invalid type for {field}: expected {expected_type}, got {type(data[field])}")
        
        # Validate confidence score range
        if 'confidence_score' in data:
            score = data['confidence_score']
            if not (self.quality_thresholds['minimum_confidence'] <= score <= self.quality_thresholds['maximum_confidence']):
                errors.append(f"Confidence score out of range: {score}")
        
        # Validate risk level
        if 'risk_level' in data:
            if data['risk_level'] not in self.quality_thresholds['valid_risk_levels']:
                errors.append(f"Invalid risk level: {data['risk_level']}")
        
        # Validate signal validity structure
        if 'signal_validity' in data:
            validity = data['signal_validity']
            if not isinstance(validity, dict):
                errors.append("signal_validity must be a dictionary")
            elif 'is_valid' not in validity:
                errors.append("signal_validity missing 'is_valid' field")
        
        # Validate justification structure
        if 'justification' in data:
            just = data['justification']
            if not isinstance(just, dict):
                errors.append("justification must be a dictionary")
            else:
                for level in ['novice', 'intermediate', 'expert']:
                    if level not in just:
                        errors.append(f"justification missing '{level}' level")
                    elif level in just:
                        min_length = self.quality_thresholds['minimum_justification_length'][level]
                        if len(just[level]) < min_length:
                            errors.append(f"justification['{level}'] too short (min {min_length} chars)")
        
        return errors
    
    def _extract_fields(self, data: Dict[str, Any], raw_response: str) -> ProcessedResponse:
        """Extract and process response fields."""
        return ProcessedResponse(
            confidence_score=data.get('confidence_score', 50.0),
            risk_level=data.get('risk_level', 'medium'),
            signal_validity=data.get('signal_validity', {'is_valid': False, 'reasons': []}),
            optimizations=data.get('optimizations', {}),
            risk_factors=data.get('risk_factors', []),
            recommendations=data.get('recommendations', []),
            justification=data.get('justification', {
                'novice': 'Analysis unavailable',
                'intermediate': 'Analysis unavailable',
                'expert': 'Analysis unavailable'
            }),
            raw_response=raw_response,
            processing_status='pending',
            validation_errors=[],
            metadata={
                'processed_at': datetime.now().isoformat(),
                'response_length': len(raw_response)
            }
        )
    
    def _score_quality(self, response: ProcessedResponse) -> float:
        """Score response quality from 0-100."""
        score = 0
        max_score = 100
        
        # Completeness (30 points)
        if response.confidence_score != 50.0:  # Not default
            score += 10
        if response.risk_level in self.quality_thresholds['valid_risk_levels']:
            score += 10
        if response.signal_validity.get('reasons'):
            score += 10
        
        # Detail level (30 points)
        if len(response.risk_factors) > 0:
            score += min(len(response.risk_factors) * 3, 10)
        if len(response.recommendations) > 0:
            score += min(len(response.recommendations) * 3, 10)
        if response.optimizations:
            score += 10
        
        # Justification quality (30 points)
        just = response.justification
        if len(just.get('novice', '')) >= 50:
            score += 10
        if len(just.get('intermediate', '')) >= 200:
            score += 10
        if len(just.get('expert', '')) >= 500:
            score += 10
        
        # Consistency (10 points)
        if not response.validation_errors:
            score += 10
        
        return min(score, max_score)
    
    def _cache_response(self, key: str, response: ProcessedResponse):
        """Cache processed response."""
        import hashlib
        
        # Generate cache key
        cache_key = hashlib.md5(key.encode()).hexdigest()
        
        # Store with timestamp
        self.response_cache[cache_key] = {
            'response': response,
            'timestamp': datetime.now()
        }
        
        # Clean old cache entries
        cutoff = datetime.now().timestamp() - self.cache_ttl
        self.response_cache = {
            k: v for k, v in self.response_cache.items()
            if v['timestamp'].timestamp() > cutoff
        }
    
    def post_process(self, response: ProcessedResponse) -> ProcessedResponse:
        """Apply post-processing pipeline to response."""
        # Clean and format risk factors
        if response.risk_factors:
            response.risk_factors = [self._format_risk_factor(rf) for rf in response.risk_factors]
        
        # Clean and format recommendations
        if response.recommendations:
            response.recommendations = [self._format_recommendation(r) for r in response.recommendations]
        
        # Ensure optimization prices are valid
        if response.optimizations:
            response.optimizations = self._validate_optimizations(response.optimizations)
        
        # Format justification
        response.justification = self._format_justification(response.justification)
        
        return response
    
    def _format_risk_factor(self, risk_factor: str) -> str:
        """Format risk factor for presentation."""
        # Convert snake_case to readable format
        formatted = risk_factor.replace('_', ' ').title()
        
        # Add emojis for visual clarity
        emoji_map = {
            'High Volatility': '📊',
            'Low Liquidity': '💧',
            'Manipulation Risk': '⚠️',
            'Price Divergence': '📉',
            'Opposite Signals': '🔄'
        }
        
        if formatted in emoji_map:
            formatted = f"{emoji_map[formatted]} {formatted}"
        
        return formatted
    
    def _format_recommendation(self, recommendation: str) -> str:
        """Format recommendation for presentation."""
        # Capitalize first letter
        if recommendation:
            recommendation = recommendation[0].upper() + recommendation[1:]
        
        # Ensure ends with period
        if recommendation and not recommendation.endswith('.'):
            recommendation += '.'
        
        return recommendation
    
    def _validate_optimizations(self, optimizations: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and clean optimization suggestions."""
        validated = {}
        
        # Validate entry price
        if 'entry_price' in optimizations:
            try:
                price = float(optimizations['entry_price'])
                if price > 0:
                    validated['entry_price'] = price
            except (ValueError, TypeError):
                pass
        
        # Validate stop loss
        if 'stop_loss' in optimizations:
            try:
                sl = float(optimizations['stop_loss'])
                if sl > 0:
                    validated['stop_loss'] = sl
            except (ValueError, TypeError):
                pass
        
        # Validate take profits
        if 'take_profits' in optimizations:
            tps = optimizations['take_profits']
            if isinstance(tps, list):
                validated_tps = []
                for tp in tps:
                    try:
                        tp_price = float(tp)
                        if tp_price > 0:
                            validated_tps.append(tp_price)
                    except (ValueError, TypeError):
                        pass
                if validated_tps:
                    validated['take_profits'] = validated_tps
        
        return validated
    
    def _format_justification(self, justification: Dict[str, str]) -> Dict[str, str]:
        """Format justification text for better readability."""
        formatted = {}
        
        for level in ['novice', 'intermediate', 'expert']:
            text = justification.get(level, '')
            
            # Clean up text
            text = text.strip()
            
            # Ensure proper capitalization
            if text and text[0].islower():
                text = text[0].upper() + text[1:]
            
            # Add line breaks for readability (expert level)
            if level == 'expert' and len(text) > 500:
                # Add paragraph breaks every ~300 characters at sentence boundaries
                sentences = text.split('. ')
                formatted_text = ''
                current_paragraph = ''
                
                for sentence in sentences:
                    if len(current_paragraph) + len(sentence) > 300:
                        if current_paragraph:
                            formatted_text += current_paragraph + '.\n\n'
                            current_paragraph = sentence
                    else:
                        if current_paragraph:
                            current_paragraph += '. ' + sentence
                        else:
                            current_paragraph = sentence
                
                if current_paragraph:
                    formatted_text += current_paragraph
                    if not formatted_text.endswith('.'):
                        formatted_text += '.'
                
                text = formatted_text
            
            formatted[level] = text
        
        return formatted
    
    def audit_response(self, response: ProcessedResponse) -> Dict[str, Any]:
        """Generate audit log for response."""
        return {
            'timestamp': datetime.now().isoformat(),
            'processing_status': response.processing_status,
            'validation_errors': response.validation_errors,
            'quality_score': response.metadata.get('quality_score', 0),
            'confidence_score': response.confidence_score,
            'risk_level': response.risk_level,
            'signal_valid': response.signal_validity.get('is_valid', False),
            'has_optimizations': bool(response.optimizations),
            'risk_factors_count': len(response.risk_factors),
            'recommendations_count': len(response.recommendations),
            'response_length': response.metadata.get('response_length', 0)
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get processor statistics."""
        avg_quality = sum(self.quality_scores) / len(self.quality_scores) if self.quality_scores else 0
        
        return {
            'total_processed': len(self.quality_scores),
            'avg_quality_score': round(avg_quality, 2),
            'cache_size': len(self.response_cache),
            'quality_distribution': {
                'excellent': sum(1 for s in self.quality_scores if s >= 80),
                'good': sum(1 for s in self.quality_scores if 60 <= s < 80),
                'fair': sum(1 for s in self.quality_scores if 40 <= s < 60),
                'poor': sum(1 for s in self.quality_scores if s < 40)
            }
        }
