"""
ElevenLabs TTS Engine - Cloud-based high-quality text-to-speech.
Requires an ElevenLabs API key.
"""
import os
import requests
import json
from pathlib import Path
from uuid import uuid4
import math
import re
import config

ELEVENLABS_AVAILABLE = True  # Available if user provides API key


class ElevenLabsTTS:
    """ElevenLabs TTS engine for high-quality voice synthesis."""

    def __init__(self):
        self.api_key = config.ELEVENLABS_API_KEY or None
        self.base_url = "https://api.elevenlabs.io/v1"
        self._voices_cache = None

    def set_api_key(self, api_key: str):
        """Set the ElevenLabs API key."""
        self.api_key = api_key

    @property
    def available(self) -> bool:
        """Check if ElevenLabs is available (has API key)."""
        return bool(self.api_key)

    def generate_speech(self, text: str, output_path: str,
                        voice: str = None, model: str = "eleven_multilingual_v2",
                        stability: float = 0.5, similarity_boost: float = 0.75,
                        previous_text: str = None, next_text: str = None, seed: int = None) -> dict:
        """
        Generate speech audio using ElevenLabs API.

        Args:
            text: Text to synthesize
            output_path: Path to save the audio file
            voice: ElevenLabs voice ID
            model: Model ID (eleven_monolingual_v1, eleven_multilingual_v2, etc.)
            stability: Stability setting (0.0-1.0)
            similarity_boost: Clarity setting (0.0-1.0)

        Returns:
            dict with audio_path, duration
        """
        if not self.api_key:
            raise RuntimeError("ElevenLabs API key not set. Please configure your API key in settings.")
        if not isinstance(model, str) or not re.fullmatch(r"[a-z0-9_]{1,80}", model):
            raise ValueError("Invalid ElevenLabs model")
        for value in (stability, similarity_boost):
            if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError("Voice settings must be between 0 and 1")
        if model == "eleven_v3":
            stability = min((0., .5, 1.), key=lambda v: abs(v - stability))
        if seed is not None and (type(seed) is not int or not 0 <= seed <= 4294967295):
            raise ValueError("Invalid synthesis seed")

        if not voice:
            # Default to first available voice
            voices = self.get_voices()
            if not voices:
                raise RuntimeError("No ElevenLabs voices available")
            voice = voices[0]['voice_id']
        if not isinstance(voice, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,120}", voice):
            raise ValueError("Invalid ElevenLabs voice ID")

        # ElevenLabs API endpoint
        url = f"{self.base_url}/text-to-speech/{voice}"

        headers = {
            "Accept": "audio/mpeg",
            "Content-Type": "application/json",
            "xi-api-key": self.api_key
        }

        data = {
            "text": text,
            "model_id": model,
            "voice_settings": {
                "stability": stability,
                "similarity_boost": similarity_boost
            }
        }
        # v4 has no style/speed sliders or SSML. Performance is expressed in text tags.
        for field, value in (("previous_text", previous_text), ("next_text", next_text), ("seed", seed)):
            if value is not None:
                data[field] = value

        try:
            response = requests.post(url, json=data, headers=headers, params={"output_format": "mp3_44100_128"}, timeout=90)
            response.raise_for_status()

            # Ensure output directory exists
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)

            # Save audio
            mp3_path = output_path.rsplit('.', 1)[0] + '.mp3'
            temporary = Path(mp3_path).with_name(f".{uuid4().hex}.mp3")
            try:
                temporary.write_bytes(response.content)
                duration = self._get_audio_duration(str(temporary))
                temporary.replace(mp3_path)
            finally:
                temporary.unlink(missing_ok=True)

            return {
                "audio_path": mp3_path,
                "duration": duration,
            }

        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 401:
                raise RuntimeError("Invalid ElevenLabs API key")
            elif e.response.status_code == 403:
                raise RuntimeError("ElevenLabs API quota exceeded or access denied")
            elif e.response.status_code == 422:
                raise RuntimeError(f"Invalid request to ElevenLabs API: {e.response.text}")
            else:
                raise RuntimeError(f"ElevenLabs API error: {e}")
        except requests.exceptions.Timeout:
            raise RuntimeError("ElevenLabs API request timed out")
        except Exception as e:
            raise RuntimeError(f"ElevenLabs TTS error: {e}")

    def get_voices(self, force_refresh: bool = False) -> list[dict]:
        """
        Get available ElevenLabs voices.
        
        Returns list of voice dicts with: voice_id, name, preview_url, labels, etc.
        """
        if not self.api_key:
            return []

        # Use cache unless force refresh
        if self._voices_cache is not None and not force_refresh:
            return self._voices_cache

        url = f"{self.base_url}/voices"
        headers = {"xi-api-key": self.api_key}

        try:
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            data = response.json()

            voices = []
            for v in data.get('voices', []):
                voices.append({
                    "voice_id": v['voice_id'],
                    "name": v['name'],
                    "preview_url": v.get('preview_url'),
                    "category": v.get('category', 'generated'),
                    "labels": v.get('labels', {}),
                    "description": v.get('description', ''),
                    "engine": "elevenlabs",
                })

            self._voices_cache = voices
            return voices

        except Exception as e:
            print(f"[ElevenLabs] Error fetching voices: {e}")
            return []

    def _get_audio_duration(self, audio_path: str) -> float:
        """Get audio duration in seconds."""
        from pipeline.timeline_director import audio_duration
        return audio_duration(audio_path)

    def test_api_key(self) -> tuple[bool, str]:
        """
        Test if the API key is valid.
        
        Returns:
            (success: bool, message: str)
        """
        if not self.api_key:
            return False, "No API key provided"

        try:
            voices = self.get_voices(force_refresh=True)
            if voices:
                return True, f"API key valid. {len(voices)} voices available."
            else:
                return False, "API key valid but no voices found"
        except Exception as e:
            return False, str(e)


# Singleton
_elevenlabs_tts = None


def get_elevenlabs_tts() -> ElevenLabsTTS:
    """Get the global ElevenLabs TTS instance."""
    global _elevenlabs_tts
    if _elevenlabs_tts is None:
        _elevenlabs_tts = ElevenLabsTTS()
    return _elevenlabs_tts
