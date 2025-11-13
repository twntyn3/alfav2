"""Text processing and normalization utilities for LLM-ready data preparation."""
import re
import unicodedata
from typing import Optional


def normalize_whitespace(text: str) -> str:
    """Normalize whitespace: collapse multiple spaces, preserve newlines."""
    # replace tabs with spaces
    text = text.replace("\t", " ")
    # collapse multiple spaces but preserve newlines
    lines = text.split("\n")
    normalized_lines = []
    for line in lines:
        # collapse multiple spaces within line
        line = re.sub(r" +", " ", line)
        normalized_lines.append(line.strip())
    return "\n".join(normalized_lines)


def remove_control_characters(text: str) -> str:
    """Remove control characters except newlines, tabs, and carriage returns."""
    # keep newlines (\n), tabs (\t), carriage returns (\r)
    allowed = {"\n", "\t", "\r"}
    result = []
    for char in text:
        if unicodedata.category(char)[0] == "C":  # control character
            if char not in allowed:
                continue
        result.append(char)
    return "".join(result)


def normalize_unicode(text: str) -> str:
    """Normalize unicode characters (NFKC normalization)."""
    return unicodedata.normalize("NFKC", text)


def clean_text(text: str, preserve_structure: bool = True) -> str:
    """
    Comprehensive text cleaning for LLM preparation.
    
    Args:
        text: Input text
        preserve_structure: If True, preserve paragraph structure (newlines)
        
    Returns:
        Cleaned text ready for LLM processing
    """
    if not text:
        return ""
    
    # normalize unicode first
    text = normalize_unicode(text)
    
    # remove control characters (except newlines/tabs if preserving structure)
    text = remove_control_characters(text)
    
    # normalize whitespace
    text = normalize_whitespace(text)
    
    # remove zero-width characters
    text = re.sub(r"[\u200b-\u200f\u2028-\u202f\u2060-\u206f\ufeff]", "", text)
    
    # remove excessive newlines (more than 2 consecutive)
    if preserve_structure:
        text = re.sub(r"\n{3,}", "\n\n", text)
    else:
        text = re.sub(r"\n+", " ", text)
    
    return text.strip()


def extract_and_clean_title(title: str, normalize_for_search: bool = True, normalization_mode: str = "smart") -> str:
    """
    Extract and clean title, handling edge cases.
    
    Args:
        title: Input title
        normalize_for_search: Apply retrieval normalization
        normalization_mode: "smart", "letters_numbers", or "aggressive"
    """
    if not title:
        return ""
    
    title = clean_text(title, preserve_structure=False)
    # remove excessive punctuation at end
    title = re.sub(r"[.!?]{2,}$", ".", title)
    
    # Apply retrieval normalization if requested (to match text normalization)
    if normalize_for_search:
        title = normalize_for_retrieval(title, mode=normalization_mode)
    
    return title.strip()


def remove_common_footers(text: str) -> str:
    """
    Remove common footer patterns that appear in multiple documents.
    
    This helps reduce noise and improve retrieval quality by removing
    repetitive legal text, copyright notices, etc.
    """
    if not text:
        return text
    
    # Common footer patterns (banking domain)
    footer_patterns = [
        # Copyright and legal info
        r"©\s*\d{4}[-\d]*\s*АО\s*«Альфа-Банк».*?$",
        r"АО\s*«Альфа-Банк»\s*является\s*оператором\s*по\s*обработке\s*персональных\s*данных.*?$",
        r"Генеральная\s*лицензия\s*Банка\s*России.*?$",
        r"Центр\s*раскрытия\s*корпоративной\s*информации.*?$",
        r"Информация\s*профессионального\s*участника\s*рынка\s*ценных\s*бумаг.*?$",
        r"Ул\.\s*Каланчевская.*?Москва.*?$",
        r"АО\s*«Альфа-Банк»\s*использует\s*файлы\s*«cookie».*?$",
        # Generic patterns
        r"©\s*\d{4}[-\d]*.*?$",  # Generic copyright
        r"Все\s*права\s*защищены.*?$",
    ]
    
    lines = text.split("\n")
    cleaned_lines = []
    
    for line in lines:
        line_stripped = line.strip()
        if not line_stripped:
            cleaned_lines.append(line)
            continue
        
        # Check if line matches any footer pattern
        is_footer = False
        for pattern in footer_patterns:
            if re.search(pattern, line_stripped, re.IGNORECASE | re.MULTILINE):
                is_footer = True
                break
        
        if not is_footer:
            cleaned_lines.append(line)
    
    result = "\n".join(cleaned_lines)
    # Remove trailing empty lines
    result = result.rstrip()
    
    return result


def normalize_for_retrieval(text: str, mode: str = "letters_numbers") -> str:
    """
    Normalize text for better retrieval: lowercase + optional symbol removal.
    
    Args:
        text: Input text
        mode: 
            - "letters_numbers" (default): lowercase + letters + digits + spaces
            - "smart": lowercase + letters + digits + important symbols (|, %, ., ,, -)
            - "aggressive": only letters + spaces (loses numbers - NOT recommended!)
        
    Returns:
        Normalized text optimized for retrieval
    """
    if not text:
        return ""
    
    text = text.lower()
    
    if mode == "aggressive":
        # Only letters and spaces (loses numbers - NOT recommended for banking!)
        text = re.sub(r'[^а-яёa-z\s]', ' ', text)
    elif mode == "smart":
        # Preserve digits and important symbols for banking domain
        # Important symbols: | (tables), % (percentages), . , (decimals), - (ranges)
        text = re.sub(r'[^а-яёa-z0-9\s|%.,\-]', ' ', text)
    else:  # letters_numbers mode (default)
        # Lowercase + letters + digits + spaces only
        text = re.sub(r'[^а-яёa-z0-9\s]', ' ', text)
    
    # Normalize multiple spaces
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()


def extract_and_clean_text(text: str, normalize_for_search: bool = True, normalization_mode: str = "letters_numbers") -> str:
    """
    Extract and clean main text, preserving structure.
    
    Args:
        text: Input text
        normalize_for_search: If True, apply retrieval normalization
        normalization_mode: "letters_numbers" (default), "smart", or "aggressive"
    """
    if not text:
        return ""
    
    text = clean_text(text, preserve_structure=True)
    # Remove common footers to reduce noise
    text = remove_common_footers(text)
    
    # Apply retrieval normalization if requested
    if normalize_for_search:
        text = normalize_for_retrieval(text, mode=normalization_mode)
    
    return text


def merge_short_paragraphs(text: str, min_length: int = 50) -> str:
    """
    Merge short paragraphs to improve chunking quality.
    
    Args:
        text: Text with paragraphs separated by newlines
        min_length: Minimum character length for standalone paragraph
        
    Returns:
        Text with short paragraphs merged
    """
    if not text:
        return ""
    
    paragraphs = text.split("\n")
    merged = []
    current = ""
    
    for para in paragraphs:
        para = para.strip()
        if not para:
            if current:
                merged.append(current)
                current = ""
            continue
        
        # if current paragraph is short, try to merge with next
        if len(current) < min_length and current:
            current += " " + para
        else:
            if current:
                merged.append(current)
            current = para
    
    if current:
        merged.append(current)
    
    return "\n".join(merged)


def validate_document(title: str, text: str) -> tuple:
    """
    Validate document quality.
    
    Returns:
        (is_valid, error_message)
    """
    if not title and not text:
        return False, "Both title and text are empty"
    
    if text and len(text.strip()) < 10:
        return False, f"Text too short: {len(text.strip())} characters"
    
    # check for excessive repetition (potential data corruption)
    if text:
        words = text.split()
        if len(words) > 0:
            unique_ratio = len(set(words)) / len(words)
            if unique_ratio < 0.1 and len(words) > 100:
                return False, f"Excessive repetition detected: {unique_ratio:.2f} unique ratio"
    
    return True, None

