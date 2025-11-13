"""Table processing and structure detection for banking domain."""
import re
from typing import List, Tuple, Dict


def detect_table(text: str) -> bool:
    """Detect if text contains a table structure."""
    # Check for markdown table format
    if re.search(r'\|.*\|', text, re.MULTILINE):
        return True
    
    # Check for tab-separated values
    lines = text.split('\n')
    if len(lines) >= 2:
        tab_count = sum(1 for line in lines[:5] if '\t' in line)
        if tab_count >= 2:
            return True
    
    # Check for multiple spaces (aligned columns)
    space_aligned = sum(1 for line in lines[:5] if re.search(r'  +', line))
    if space_aligned >= 2:
        return True
    
    return False


def extract_table_structure(text: str) -> Tuple[str, bool]:
    """
    Extract and normalize table structure.
    
    Returns:
        (normalized_text, is_table) - normalized text with preserved structure
    """
    if not detect_table(text):
        return text, False
    
    lines = text.split('\n')
    normalized_lines = []
    is_table = False
    
    for line in lines:
        line = line.strip()
        if not line:
            normalized_lines.append("")
            continue
        
        # Markdown table format
        if '|' in line:
            is_table = True
            # Clean up markdown table separators
            if re.match(r'^[\|\s\-:]+$', line):
                # Skip separator rows like |---|---|
                continue
            # Normalize table row
            cells = [cell.strip() for cell in line.split('|') if cell.strip()]
            normalized_line = " | ".join(cells)
            normalized_lines.append(normalized_line)
        # Tab-separated or space-aligned
        elif '\t' in line or re.search(r'  +', line):
            is_table = True
            # Replace multiple spaces/tabs with single space, preserve structure
            normalized_line = re.sub(r'[\t ]+', ' ', line).strip()
            normalized_lines.append(normalized_line)
        else:
            normalized_lines.append(line)
    
    normalized_text = '\n'.join(normalized_lines)
    return normalized_text, is_table


def preserve_table_in_chunk(table_text: str, chunk_text: str) -> str:
    """
    Ensure table structure is preserved when chunking.
    
    Args:
        table_text: Original table text
        chunk_text: Chunk text that may contain part of table
        
    Returns:
        Chunk text with preserved table structure
    """
    # If chunk contains table markers, ensure proper formatting
    if '|' in chunk_text:
        lines = chunk_text.split('\n')
        formatted_lines = []
        for line in lines:
            if '|' in line:
                # Ensure proper table formatting
                cells = [cell.strip() for cell in line.split('|') if cell.strip()]
                formatted_line = " | ".join(cells)
                formatted_lines.append(formatted_line)
            else:
                formatted_lines.append(line)
        return '\n'.join(formatted_lines)
    
    return chunk_text


def extract_numeric_values(text: str) -> List[Dict[str, str]]:
    """
    Extract numeric values (amounts, rates, percentages) for enhanced retrieval.
    
    Returns:
        List of dicts with 'value', 'type', 'context'
    """
    numeric_patterns = [
        # Currency amounts: 1000 руб., $100, 100 EUR
        (r'(\d+[.,]?\d*)\s*(руб|RUB|USD|EUR|CNY|долл|евро|юань)', 'currency'),
        # Percentages: 5%, 10.5%
        (r'(\d+[.,]?\d*)%', 'percentage'),
        # Rates: 10.5%, курс 77.5
        (r'(?:курс|ставка|процент|rate)\s*[:\-]?\s*(\d+[.,]?\d*)', 'rate'),
        # Large numbers: 1000000, 1 000 000
        (r'(\d{1,3}(?:\s+\d{3})+)', 'large_number'),
        # Dates with numbers: 01.01.2024
        (r'(\d{1,2}[./]\d{1,2}[./]\d{2,4})', 'date'),
    ]
    
    extracted = []
    for pattern, num_type in numeric_patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            # Get context (20 chars before and after)
            start = max(0, match.start() - 20)
            end = min(len(text), match.end() + 20)
            context = text[start:end].strip()
            
            extracted.append({
                'value': match.group(1),
                'type': num_type,
                'context': context
            })
    
    return extracted


def enhance_query_for_numerics(query: str) -> str:
    """
    Enhance query to better match numeric values and banking terms.
    
    Adds variations of numeric patterns, banking synonyms, and query expansion
    to improve BM25 and dense retrieval matching.
    """
    enhanced = query
    
    # Extended banking domain synonyms and expansions
    banking_synonyms = {
        "счет": ["счет", "счёт", "аккаунт", "account", "банковский счет"],
        "карта": ["карта", "card", "кредитная карта", "дебетовая карта", "пластиковая карта"],
        "кредит": ["кредит", "займ", "loan", "кредитная линия", "кредитование"],
        "вклад": ["вклад", "депозит", "deposit", "сберегательный счет"],
        "платеж": ["платеж", "платёж", "оплата", "payment", "транзакция", "перевод средств"],
        "перевод": ["перевод", "transfer", "трансфер", "перечисление", "перевод денег"],
        "бик": ["бик", "bik", "банковский идентификационный код", "банковский код"],
        "реквизиты": ["реквизиты", "details", "банковские реквизиты", "платежные реквизиты"],
        "отделение": ["отделение", "офис", "банк", "branch", "office", "филиал", "банковское отделение"],
        "смс": ["смс", "sms", "сообщение", "код подтверждения", "смс-уведомление"],
        "онлайн": ["онлайн", "online", "интернет-банк", "мобильный банк", "интернет банкинг"],
        "номер": ["номер", "number", "num", "номер счета", "номер карты"],
        "узнать": ["узнать", "найти", "посмотреть", "проверить", "find", "check", "уточнить"],
        "получить": ["получить", "get", "заказать", "оформить", "выпустить"],
        "процент": ["процент", "ставка", "rate", "процентная ставка", "годовая ставка"],
        "комиссия": ["комиссия", "fee", "плата", "стоимость", "тариф"],
        "лимит": ["лимит", "limit", "ограничение", "максимум"],
        "баланс": ["баланс", "balance", "остаток", "остаток средств"],
        "выписка": ["выписка", "statement", "отчет", "история операций"],
        "кэшбэк": ["кэшбэк", "cashback", "возврат", "бонус"],
    }
    
    # Query rewriting patterns (common question reformulations)
    query_rewrites = {
        "как": ["как", "способ", "метод", "инструкция"],
        "где": ["где", "место", "адрес", "локация"],
        "когда": ["когда", "время", "срок", "дата"],
        "сколько": ["сколько", "сумма", "размер", "объем"],
        "можно": ["можно", "возможно", "разрешено", "доступно"],
        "нужно": ["нужно", "необходимо", "требуется", "надо"],
    }
    
    # Add synonyms for key banking terms
    query_lower = query.lower()
    added_synonyms = set()
    
    for key, synonyms in banking_synonyms.items():
        if key in query_lower:
            for synonym in synonyms[:3]:  # Add first 3 synonyms
                if synonym not in query_lower and synonym not in added_synonyms:
                    enhanced += f" {synonym}"
                    added_synonyms.add(synonym)
    
    # Add query rewriting patterns
    for pattern, rewrites in query_rewrites.items():
        if pattern in query_lower:
            for rewrite in rewrites[:2]:  # Add first 2 rewrites
                if rewrite not in query_lower and rewrite not in added_synonyms:
                    enhanced += f" {rewrite}"
                    added_synonyms.add(rewrite)
    
    # Extract numbers from query and add variations
    numbers = re.findall(r'\d+[.,]?\d*', query)
    for num in numbers:
        # Add variations: with/without spaces, with/without decimal
        if '.' in num or ',' in num:
            # Add integer version
            int_version = re.sub(r'[.,]\d+', '', num)
            if int_version and int_version not in enhanced:
                enhanced += f" {int_version}"
        else:
            # Add decimal versions
            enhanced += f" {num}.0 {num},0"
    
    # Add common banking query patterns
    if any(word in query_lower for word in ["где", "как", "какой", "что"]):
        # Add "информация" and "помощь" for informational queries
        if "информация" not in enhanced.lower():
            enhanced += " информация"
        if "условия" not in enhanced.lower() and any(w in query_lower for w in ["кредит", "вклад", "карта"]):
            enhanced += " условия"
    
    # Remove extra spaces and limit length
    enhanced = re.sub(r'\s+', ' ', enhanced).strip()
    # Limit to reasonable length (keep original + ~100 chars expansion)
    if len(enhanced) > len(query) + 150:
        # Keep original query + first 30 words of expansion
        words = enhanced.split()
        original_words = query.split()
        max_expansion_words = 30
        enhanced = " ".join(original_words + words[len(original_words):len(original_words) + max_expansion_words])
    
    return enhanced

