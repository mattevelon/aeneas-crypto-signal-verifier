"""
Secret rotation mechanism for API credentials.
"""

import os
import json
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from pathlib import Path
import hashlib
import secrets
from cryptography.fernet import Fernet
import structlog

from src.config.settings import settings

logger = structlog.get_logger()


class SecretRotationManager:
    """Manages secret rotation and versioning."""
    
    def __init__(self, rotation_days: int = 90):
        """
        Initialize secret rotation manager.
        
        Args:
            rotation_days: Days before rotation is required
        """
        self.rotation_days = rotation_days
        self.secrets_file = Path(".secrets/rotation_state.json")
        self.secrets_file.parent.mkdir(exist_ok=True)
        self.encryption_key = self._get_or_create_key()
        self.cipher = Fernet(self.encryption_key)
        self.rotation_state = self._load_rotation_state()
    
    def _get_or_create_key(self) -> bytes:
        """Get or create encryption key for secrets file."""
        key_file = Path(".secrets/encryption.key")
        key_file.parent.mkdir(exist_ok=True)
        
        if key_file.exists():
            return key_file.read_bytes()
        
        key = Fernet.generate_key()
        key_file.write_bytes(key)
        key_file.chmod(0o600)  # Read/write for owner only
        return key
    
    def _load_rotation_state(self) -> Dict[str, Any]:
        """Load rotation state from encrypted file."""
        if not self.secrets_file.exists():
            return {}
        
        try:
            encrypted_data = self.secrets_file.read_bytes()
            decrypted_data = self.cipher.decrypt(encrypted_data)
            return json.loads(decrypted_data.decode())
        except Exception as e:
            logger.error(f"Failed to load rotation state: {e}")
            return {}
    
    def _save_rotation_state(self):
        """Save rotation state to encrypted file."""
        try:
            data = json.dumps(self.rotation_state, indent=2, default=str)
            encrypted_data = self.cipher.encrypt(data.encode())
            self.secrets_file.write_bytes(encrypted_data)
            self.secrets_file.chmod(0o600)
        except Exception as e:
            logger.error(f"Failed to save rotation state: {e}")
    
    def check_rotation_needed(self, secret_name: str) -> bool:
        """
        Check if a secret needs rotation.
        
        Args:
            secret_name: Name of the secret to check
            
        Returns:
            True if rotation is needed
        """
        if secret_name not in self.rotation_state:
            return True
        
        last_rotation = datetime.fromisoformat(
            self.rotation_state[secret_name]["last_rotation"]
        )
        rotation_due = last_rotation + timedelta(days=self.rotation_days)
        
        return datetime.now() > rotation_due
    
    def get_secrets_needing_rotation(self) -> List[str]:
        """Get list of secrets that need rotation."""
        secrets_to_rotate = []
        
        # Check all configured secrets
        secret_configs = {
            "telegram_api": settings.telegram_api_id is not None,
            "llm_api_key": settings.llm_api_key is not None,
            "binance_api": settings.binance_api_key is not None,
            "kucoin_api": settings.kucoin_api_key is not None,
            "jwt_secret": True,  # Always check JWT secret
        }
        
        for secret_name, is_configured in secret_configs.items():
            if is_configured and self.check_rotation_needed(secret_name):
                secrets_to_rotate.append(secret_name)
        
        return secrets_to_rotate
    
    def mark_rotated(self, secret_name: str, metadata: Optional[Dict] = None):
        """
        Mark a secret as rotated.
        
        Args:
            secret_name: Name of the rotated secret
            metadata: Optional metadata about the rotation
        """
        self.rotation_state[secret_name] = {
            "last_rotation": datetime.now().isoformat(),
            "rotation_count": self.rotation_state.get(secret_name, {}).get("rotation_count", 0) + 1,
            "metadata": metadata or {}
        }
        self._save_rotation_state()
        logger.info(f"Secret '{secret_name}' marked as rotated")
    
    def generate_jwt_secret(self) -> str:
        """Generate a new secure JWT secret."""
        return secrets.token_urlsafe(64)
    
    def hash_api_key(self, api_key: str) -> str:
        """Hash an API key for comparison without storing plaintext."""
        return hashlib.sha256(api_key.encode()).hexdigest()
    
    async def rotate_jwt_secret(self) -> bool:
        """
        Rotate JWT secret key.
        
        Returns:
            True if rotation successful
        """
        try:
            new_secret = self.generate_jwt_secret()
            
            # Update .env file
            env_file = Path(".env")
            if env_file.exists():
                lines = env_file.read_text().splitlines()
                for i, line in enumerate(lines):
                    if line.startswith("JWT_SECRET_KEY="):
                        lines[i] = f"JWT_SECRET_KEY={new_secret}"
                        break
                else:
                    lines.append(f"JWT_SECRET_KEY={new_secret}")
                
                env_file.write_text("\n".join(lines) + "\n")
                
                self.mark_rotated("jwt_secret", {
                    "key_hash": self.hash_api_key(new_secret)[:8]  # Store partial hash
                })
                
                logger.info("JWT secret rotated successfully")
                return True
            
            logger.warning(".env file not found, cannot rotate JWT secret")
            return False
            
        except Exception as e:
            logger.error(f"Failed to rotate JWT secret: {e}")
            return False
    
    async def check_and_notify_rotations(self) -> Dict[str, List[str]]:
        """
        Check for needed rotations and return notification.
        
        Returns:
            Dictionary with rotation status
        """
        secrets_to_rotate = self.get_secrets_needing_rotation()
        
        result = {
            "needs_rotation": secrets_to_rotate,
            "rotation_status": {}
        }
        
        for secret_name in self.rotation_state:
            last_rotation = datetime.fromisoformat(
                self.rotation_state[secret_name]["last_rotation"]
            )
            days_since = (datetime.now() - last_rotation).days
            result["rotation_status"][secret_name] = {
                "last_rotation": last_rotation.isoformat(),
                "days_since_rotation": days_since,
                "rotation_due": days_since >= self.rotation_days
            }
        
        if secrets_to_rotate:
            logger.warning(
                f"Secrets needing rotation: {', '.join(secrets_to_rotate)}"
            )
        
        return result
    
    def get_rotation_schedule(self) -> Dict[str, Any]:
        """Get rotation schedule for all secrets."""
        schedule = {}
        
        for secret_name, state in self.rotation_state.items():
            last_rotation = datetime.fromisoformat(state["last_rotation"])
            next_rotation = last_rotation + timedelta(days=self.rotation_days)
            days_until = (next_rotation - datetime.now()).days
            
            schedule[secret_name] = {
                "last_rotation": last_rotation.isoformat(),
                "next_rotation": next_rotation.isoformat(),
                "days_until_rotation": max(0, days_until),
                "overdue": days_until < 0,
                "rotation_count": state.get("rotation_count", 1)
            }
        
        return schedule


# Global instance
secret_rotation_manager = SecretRotationManager()


async def check_secret_rotations():
    """Check and report on secret rotations needed."""
    return await secret_rotation_manager.check_and_notify_rotations()


async def rotate_jwt():
    """Rotate the JWT secret."""
    return await secret_rotation_manager.rotate_jwt_secret()


async def get_rotation_schedule():
    """Get the rotation schedule for all secrets."""
    return secret_rotation_manager.get_rotation_schedule()


# Background task for periodic rotation checks
async def rotation_monitor():
    """Background task to monitor secret rotation needs."""
    while True:
        try:
            await check_secret_rotations()
            await asyncio.sleep(86400)  # Check daily
        except Exception as e:
            logger.error(f"Rotation monitor error: {e}")
            await asyncio.sleep(3600)  # Retry in 1 hour on error
