"""Token Optimizer for managing LLM context window efficiently."""

import json
from typing import Dict, Any, List, Optional, Tuple
import logging
import tiktoken
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class TokenMetrics:
    """Token usage metrics."""
    total_tokens: int
    system_tokens: int
    user_tokens: int
    context_tokens: int
    remaining_budget: int
    sections: Dict[str, int]


class TokenOptimizer:
    """
    Optimizes token usage for LLM context windows.
    Implements priority-based truncation and compression strategies.
    """
    
    def __init__(self, max_tokens: int = 8000):
        """
        Initialize token optimizer.
        
        Args:
            max_tokens: Maximum token budget
        """
        self.max_tokens = max_tokens
        self.reserved_tokens = 1000  # Reserve for response
        self.available_tokens = max_tokens - self.reserved_tokens
        
        # Token estimation models
        try:
            self.encoding = tiktoken.encoding_for_model("gpt-4")
        except:
            self.encoding = tiktoken.get_encoding("cl100k_base")
        
        # Priority weights for different sections
        self.section_priorities = {
            'signal_details': 1.0,      # Highest priority
            'technical_indicators': 0.9,
            'market_context': 0.8,
            'cross_channel_data': 0.7,
            'historical_data': 0.6,     # Lowest priority
        }
        
        # Compression strategies
        self.compression_rules = {
            'remove_duplicates': True,
            'summarize_arrays': True,
            'truncate_strings': True,
            'remove_nulls': True,
            'round_numbers': True
        }
        
        logger.info(f"Initialized TokenOptimizer with {max_tokens} token budget")
    
    def optimize_context(self, context: Dict[str, Any]) -> Tuple[Dict[str, Any], TokenMetrics]:
        """
        Optimize context to fit within token budget.
        
        Args:
            context: Full context dictionary
            
        Returns:
            Tuple of (optimized_context, token_metrics)
        """
        # Calculate initial token usage
        initial_metrics = self._calculate_tokens(context)
        
        if initial_metrics.total_tokens <= self.available_tokens:
            return context, initial_metrics
        
        # Apply optimization strategies
        optimized = self._apply_optimization(context, initial_metrics)
        
        # Calculate final metrics
        final_metrics = self._calculate_tokens(optimized)
        
        return optimized, final_metrics
    
    def _calculate_tokens(self, data: Any) -> TokenMetrics:
        """Calculate token usage for data."""
        if isinstance(data, dict):
            # Calculate tokens per section
            sections = {}
            total = 0
            
            for key, value in data.items():
                section_tokens = self._count_tokens(json.dumps(value, default=str))
                sections[key] = section_tokens
                total += section_tokens
            
            return TokenMetrics(
                total_tokens=total,
                system_tokens=sections.get('system', 0),
                user_tokens=sections.get('user', 0),
                context_tokens=total - sections.get('system', 0) - sections.get('user', 0),
                remaining_budget=self.available_tokens - total,
                sections=sections
            )
        else:
            tokens = self._count_tokens(str(data))
            return TokenMetrics(
                total_tokens=tokens,
                system_tokens=0,
                user_tokens=tokens,
                context_tokens=0,
                remaining_budget=self.available_tokens - tokens,
                sections={}
            )
    
    def _count_tokens(self, text: str) -> int:
        """Count tokens in text using tiktoken."""
        try:
            return len(self.encoding.encode(text))
        except:
            # Fallback to rough estimation
            return len(text) // 4
    
    def _apply_optimization(self, context: Dict[str, Any], metrics: TokenMetrics) -> Dict[str, Any]:
        """Apply optimization strategies to reduce token usage."""
        optimized = context.copy()
        
        # Calculate how many tokens need to be removed
        excess_tokens = metrics.total_tokens - self.available_tokens
        
        # Apply compression first
        if self.compression_rules['remove_duplicates']:
            optimized = self._remove_duplicates(optimized)
        
        if self.compression_rules['remove_nulls']:
            optimized = self._remove_nulls(optimized)
        
        if self.compression_rules['round_numbers']:
            optimized = self._round_numbers(optimized)
        
        # Check if compression was sufficient
        current_tokens = self._count_tokens(json.dumps(optimized, default=str))
        
        if current_tokens <= self.available_tokens:
            return optimized
        
        # Apply priority-based truncation
        optimized = self._priority_truncation(optimized, metrics)
        
        # Final aggressive truncation if needed
        current_tokens = self._count_tokens(json.dumps(optimized, default=str))
        if current_tokens > self.available_tokens:
            optimized = self._aggressive_truncation(optimized)
        
        return optimized
    
    def _remove_duplicates(self, data: Any) -> Any:
        """Remove duplicate information from data."""
        if isinstance(data, dict):
            cleaned = {}
            seen_values = set()
            
            for key, value in data.items():
                if isinstance(value, (dict, list)):
                    cleaned[key] = self._remove_duplicates(value)
                else:
                    # Skip duplicate scalar values
                    value_str = str(value)
                    if len(value_str) > 50 and value_str not in seen_values:
                        seen_values.add(value_str)
                        cleaned[key] = value
                    elif len(value_str) <= 50:
                        cleaned[key] = value
            
            return cleaned
        
        elif isinstance(data, list):
            # Remove duplicate items from lists
            seen = []
            for item in data:
                if item not in seen:
                    seen.append(item)
            return seen
        
        return data
    
    def _remove_nulls(self, data: Any) -> Any:
        """Remove null/empty values from data."""
        if isinstance(data, dict):
            return {
                k: self._remove_nulls(v)
                for k, v in data.items()
                if v is not None and v != "" and v != []
            }
        elif isinstance(data, list):
            return [self._remove_nulls(item) for item in data if item is not None]
        return data
    
    def _round_numbers(self, data: Any) -> Any:
        """Round numbers to reduce token usage."""
        if isinstance(data, dict):
            return {k: self._round_numbers(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._round_numbers(item) for item in data]
        elif isinstance(data, float):
            # Round to 2 decimal places for most numbers
            if abs(data) < 1:
                return round(data, 4)
            elif abs(data) < 100:
                return round(data, 2)
            else:
                return round(data, 0)
        return data
    
    def _priority_truncation(self, context: Dict[str, Any], metrics: TokenMetrics) -> Dict[str, Any]:
        """Truncate sections based on priority."""
        optimized = context.copy()
        
        # Sort sections by priority (lowest first for removal)
        sections_by_priority = sorted(
            metrics.sections.items(),
            key=lambda x: self.section_priorities.get(x[0], 0.5)
        )
        
        current_tokens = metrics.total_tokens
        target_tokens = self.available_tokens
        
        for section_name, section_tokens in sections_by_priority:
            if current_tokens <= target_tokens:
                break
            
            if section_name in optimized:
                # Calculate how much to truncate
                reduction_needed = current_tokens - target_tokens
                reduction_ratio = min(0.5, reduction_needed / section_tokens)
                
                # Truncate section
                optimized[section_name] = self._truncate_section(
                    optimized[section_name],
                    reduction_ratio
                )
                
                # Update token count
                new_section_tokens = self._count_tokens(
                    json.dumps(optimized[section_name], default=str)
                )
                current_tokens = current_tokens - section_tokens + new_section_tokens
        
        return optimized
    
    def _truncate_section(self, section: Any, reduction_ratio: float) -> Any:
        """Truncate a section by the specified ratio."""
        if isinstance(section, dict):
            # Remove less important keys
            keys = list(section.keys())
            keys_to_keep = int(len(keys) * (1 - reduction_ratio))
            
            # Keep most important keys (assume first keys are more important)
            truncated = {}
            for i, key in enumerate(keys):
                if i < keys_to_keep:
                    truncated[key] = section[key]
                else:
                    break
            
            return truncated
        
        elif isinstance(section, list):
            # Keep only portion of list
            items_to_keep = max(1, int(len(section) * (1 - reduction_ratio)))
            return section[:items_to_keep]
        
        elif isinstance(section, str):
            # Truncate string
            chars_to_keep = max(50, int(len(section) * (1 - reduction_ratio)))
            if len(section) > chars_to_keep:
                return section[:chars_to_keep] + "..."
            return section
        
        return section
    
    def _aggressive_truncation(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Aggressively truncate to fit within budget."""
        # Keep only essential fields
        essential = ['signal_details', 'technical_indicators', 'market_context']
        
        truncated = {}
        for key in essential:
            if key in context:
                # Heavily truncate each section
                section = context[key]
                if isinstance(section, dict):
                    # Keep only first 5 keys
                    truncated[key] = dict(list(section.items())[:5])
                elif isinstance(section, list):
                    # Keep only first 3 items
                    truncated[key] = section[:3]
                elif isinstance(section, str):
                    # Keep only first 500 chars
                    truncated[key] = section[:500] + "..." if len(section) > 500 else section
                else:
                    truncated[key] = section
        
        return truncated
    
    def estimate_tokens(self, text: str) -> int:
        """
        Estimate token count for text.
        
        Args:
            text: Input text
            
        Returns:
            Estimated token count
        """
        return self._count_tokens(text)
    
    def fit_to_budget(self, text: str, budget: int) -> str:
        """
        Fit text to token budget.
        
        Args:
            text: Input text
            budget: Token budget
            
        Returns:
            Truncated text
        """
        tokens = self._count_tokens(text)
        
        if tokens <= budget:
            return text
        
        # Binary search for optimal truncation point
        left, right = 0, len(text)
        result = text
        
        while left < right:
            mid = (left + right) // 2
            truncated = text[:mid]
            
            if self._count_tokens(truncated) <= budget:
                result = truncated
                left = mid + 1
            else:
                right = mid
        
        # Add ellipsis if truncated
        if len(result) < len(text):
            result += "..."
        
        return result
    
    def summarize_data(self, data: Any, max_items: int = 5) -> Any:
        """
        Summarize data structures to reduce tokens.
        
        Args:
            data: Input data
            max_items: Maximum items to keep
            
        Returns:
            Summarized data
        """
        if isinstance(data, list) and len(data) > max_items:
            return {
                'first_items': data[:max_items],
                'total_count': len(data),
                'summary': f"Showing {max_items} of {len(data)} items"
            }
        
        elif isinstance(data, dict) and len(data) > max_items:
            items = list(data.items())[:max_items]
            return {
                **dict(items),
                '_summary': f"Showing {max_items} of {len(data)} fields"
            }
        
        return data
    
    def get_optimization_stats(self, 
                              original: Dict[str, Any],
                              optimized: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get statistics about optimization.
        
        Args:
            original: Original context
            optimized: Optimized context
            
        Returns:
            Optimization statistics
        """
        original_tokens = self._count_tokens(json.dumps(original, default=str))
        optimized_tokens = self._count_tokens(json.dumps(optimized, default=str))
        
        return {
            'original_tokens': original_tokens,
            'optimized_tokens': optimized_tokens,
            'reduction': original_tokens - optimized_tokens,
            'reduction_percentage': ((original_tokens - optimized_tokens) / original_tokens * 100) if original_tokens > 0 else 0,
            'fits_budget': optimized_tokens <= self.available_tokens,
            'remaining_budget': self.available_tokens - optimized_tokens
        }
