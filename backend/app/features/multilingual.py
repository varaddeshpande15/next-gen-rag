from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException
import logging

logger = logging.getLogger(__name__)

DetectorFactory.seed = 0

def detect_language(text: str) -> str:
    try:
        lang_code = detect(text)
        logger.info(f"Detected language: {lang_code}")
        return lang_code
    except LangDetectException:
        logger.warning("Language detection failed, defaulting to 'en'")
        return "en"

def get_language_name(code: str) -> str:
    mapping = {
        'en': 'English',
        'hi': 'Hindi',
        'ta': 'Tamil',
        'te': 'Telugu',
        'bn': 'Bengali',
        'mr': 'Marathi',
        'gu': 'Gujarati',
        'kn': 'Kannada',
        'ml': 'Malayalam',
        'pa': 'Punjabi',
        'ur': 'Urdu'
    }
    return mapping.get(code, "Unknown")
