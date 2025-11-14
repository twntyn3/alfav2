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


def extract_keywords(query: str) -> List[str]:
    """
    Extract key banking terms and important words from query.
    
    Returns:
        List of extracted keywords (significant banking terms)
    """
    if not query:
        return []
    
    # Common stop words to filter out
    stop_words = {"и", "в", "на", "с", "по", "для", "от", "до", "к", "из", "о", "об", "а", "но", "или", 
                  "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with"}
    
    # Banking domain keywords (high priority)
    banking_keywords = {
        "счет", "счёт", "карта", "кредит", "вклад", "депозит", "платеж", "платёж", "перевод",
        "реквизиты", "бик", "инн", "снилс", "отделение", "офис", "филиал", "банк", "банковский",
        "смс", "онлайн", "мобильный", "интернет", "номер", "баланс", "лимит", "комиссия",
        "процент", "ставка", "кэшбэк", "cashback", "бонус", "выписка", "statement",
        "account", "card", "loan", "credit", "deposit", "payment", "transfer", "balance",
        "fee", "rate", "branch", "office", "online", "mobile", "number", "details"
    }
    
    words = re.findall(r'\b\w+\b', query.lower())
    keywords = []
    
    for word in words:
        if len(word) >= 2 and word not in stop_words:
            # Check if word is banking keyword
            if word in banking_keywords:
                keywords.append(word)
            # Check if word contains banking terms (partial match)
            elif any(bkw in word or word in bkw for bkw in banking_keywords if len(bkw) >= 3):
                keywords.append(word)
    
    return list(set(keywords))  # Remove duplicates


def expand_abbreviations(query: str) -> str:
    """
    Expand banking abbreviations to full forms for better matching.
    
    Returns:
        Query with abbreviations expanded
    """
    abbreviations = {
        # Russian banking abbreviations
        r'\bбик\b': "бик банковский идентификационный код",
        r'\bинн\b': "инн индивидуальный номер налогоплательщика",
        r'\bснилс\b': "снилс страховой номер индивидуального лицевого счета",
        r'\bогрн\b': "огрн основной государственный регистрационный номер",
        r'\bкпп\b': "кпп код причины постановки",
        r'\bсберкнижка\b': "сберкнижка сберегательная книжка",
        r'\bмобилка\b': "мобилка мобильный банк мобильное приложение",
        r'\bонлайн\b': "онлайн интернет банкинг интернет-банк",
        r'\bдб\b': "дб дебетовая карта",
        r'\bкр\b': "кр кредитная карта",
        r'\bр/с\b': "р/с расчетный счет расчетный счёт",
        r'\bк/с\b': "к/с корреспондентский счет",
        r'\bп/с\b': "п/с платежный счет платежный счёт",
        
        # English abbreviations
        r'\bcc\b': "cc credit card",
        r'\bdc\b': "dc debit card",
        r'\batm\b': "atm банкомат автоматический банкомат",
        r'\bpos\b': "pos платежный терминал точка продаж",
        r'\bpin\b': "pin пин-код персональный идентификационный номер",
        r'\bcvv\b': "cvv код безопасности cvv код",
        r'\bcvc\b': "cvc код безопасности cvv код",
    }
    
    expanded = query
    for abbrev_pattern, expansion in abbreviations.items():
        expanded = re.sub(abbrev_pattern, lambda m: f"{m.group()} {expansion}", expanded, flags=re.IGNORECASE)
    
    return expanded


def enhance_query_for_numerics(query: str) -> str:
    """
    Enhance query to better match numeric values and banking terms.
    
    Adds variations of numeric patterns, banking synonyms, and query expansion
    to improve BM25 and dense retrieval matching.
    """
    enhanced = query.strip()
    
    # Early return for empty queries
    if not enhanced:
        return enhanced
    
    # Expand abbreviations first
    enhanced = expand_abbreviations(enhanced)
    
    # Extended banking domain synonyms and expansions (significantly expanded)
    banking_synonyms = {
        # Accounts and cards
        "счет": ["счет", "счёт", "аккаунт", "account", "банковский счет", "расчетный счет", "текущий счет"],
        "счёт": ["счёт", "счет", "аккаунт", "account", "банковский счёт", "расчётный счёт"],
        "карта": ["карта", "card", "кредитная карта", "дебетовая карта", "пластиковая карта", "банковская карта"],
        "кредитка": ["кредитка", "кредитная карта", "credit card", "кредитка"],
        "дебетка": ["дебетка", "дебетовая карта", "debit card", "дебетка"],
        
        # Loans and credits
        "кредит": ["кредит", "займ", "заём", "loan", "кредитная линия", "кредитование", "ссуда", "credit"],
        "займ": ["займ", "заём", "кредит", "loan", "ссуда"],
        "ипотека": ["ипотека", "mortgage", "ипотечный кредит", "ипотечное кредитование"],
        "автокредит": ["автокредит", "автомобильный кредит", "автокредитование", "auto loan"],
        
        # Deposits and savings
        "вклад": ["вклад", "депозит", "deposit", "сберегательный счет", "сберегательный вклад", "накопительный вклад"],
        "депозит": ["депозит", "вклад", "deposit", "сберегательный депозит"],
        "накопления": ["накопления", "сбережения", "savings", "накопительный счет"],
        
        # Payments and transfers
        "платеж": ["платеж", "платёж", "оплата", "payment", "транзакция", "перевод средств", "списание"],
        "платёж": ["платёж", "платеж", "оплата", "payment", "транзакция"],
        "перевод": ["перевод", "transfer", "трансфер", "перечисление", "перевод денег", "перевод средств"],
        "оплата": ["оплата", "платеж", "платёж", "payment", "проведение платежа"],
        "транзакция": ["транзакция", "transaction", "операция", "платеж", "перевод"],
        
        # Banking details and codes
        "бик": ["бик", "bik", "банковский идентификационный код", "банковский код", "идентификационный код"],
        "инн": ["инн", "индивидуальный номер налогоплательщика", "tax id", "налоговый номер"],
        "снилс": ["снилс", "страховой номер индивидуального лицевого счета", "пенсионный номер"],
        "реквизиты": ["реквизиты", "details", "банковские реквизиты", "платежные реквизиты", "banking details"],
        "корреспондентский счет": ["корреспондентский счет", "корсчет", "к/с", "correspondent account"],
        "расчетный счет": ["расчетный счет", "расчётный счёт", "р/с", "current account", "checking account"],
        
        # Bank locations and offices
        "отделение": ["отделение", "офис", "банк", "branch", "office", "филиал", "банковское отделение", "банковский офис"],
        "офис": ["офис", "отделение", "office", "банковский офис", "филиал"],
        "филиал": ["филиал", "branch", "отделение", "офис"],
        "банкомат": ["банкомат", "atm", "автоматический банкомат", "банкомат банка"],
        
        # Communication and notifications
        "смс": ["смс", "sms", "сообщение", "код подтверждения", "смс-уведомление", "смс код", "смс-код"],
        "уведомление": ["уведомление", "notification", "оповещение", "alert"],
        "push": ["push", "push-уведомление", "пуш", "push notification"],
        
        # Online and mobile banking
        "онлайн": ["онлайн", "online", "интернет-банк", "мобильный банк", "интернет банкинг", "интернет-банкинг"],
        "мобильный": ["мобильный", "mobile", "мобильный банк", "мобильное приложение", "мобилка"],
        "интернет-банк": ["интернет-банк", "интернет банк", "online banking", "internet banking"],
        "интернет-банкинг": ["интернет-банкинг", "интернет банкинг", "online banking"],
        
        # Numbers and identifiers
        "номер": ["номер", "number", "num", "номер счета", "номер карты", "идентификатор"],
        "номер счета": ["номер счета", "номер счёта", "account number", "счет номер"],
        "номер карты": ["номер карты", "card number", "карта номер"],
        
        # Actions
        "узнать": ["узнать", "найти", "посмотреть", "проверить", "find", "check", "уточнить", "выяснить"],
        "найти": ["найти", "find", "отыскать", "узнать", "обнаружить"],
        "получить": ["получить", "get", "заказать", "оформить", "выпустить", "приобрести"],
        "заказать": ["заказать", "order", "оформить", "request", "приобрести"],
        "оформить": ["оформить", "apply", "заказать", "выпустить", "создать"],
        "выпустить": ["выпустить", "issue", "оформить", "создать", "эмитировать"],
        
        # Rates and percentages
        "процент": ["процент", "ставка", "rate", "процентная ставка", "годовая ставка", "interest rate"],
        "ставка": ["ставка", "rate", "процентная ставка", "процент"],
        "процентная ставка": ["процентная ставка", "interest rate", "ставка процента"],
        "годовая ставка": ["годовая ставка", "annual rate", "годовой процент"],
        
        # Fees and commissions
        "комиссия": ["комиссия", "fee", "плата", "стоимость", "тариф", "сбор"],
        "плата": ["плата", "fee", "комиссия", "стоимость", "оплата"],
        "тариф": ["тариф", "tariff", "тарифный план", "план обслуживания"],
        
        # Limits and restrictions
        "лимит": ["лимит", "limit", "ограничение", "максимум", "максимальная сумма"],
        "ограничение": ["ограничение", "limit", "лимит", "restriction"],
        "максимум": ["максимум", "maximum", "максимальная сумма", "лимит"],
        
        # Balance and funds
        "баланс": ["баланс", "balance", "остаток", "остаток средств", "сумма на счете"],
        "остаток": ["остаток", "balance", "баланс", "остаток средств"],
        "средства": ["средства", "funds", "деньги", "денежные средства"],
        
        # Statements and history
        "выписка": ["выписка", "statement", "отчет", "история операций", "bank statement", "выписка по счету"],
        "история": ["история", "history", "история операций", "история транзакций"],
        "отчет": ["отчет", "report", "отчёт", "выписка"],
        
        # Bonuses and rewards
        "кэшбэк": ["кэшбэк", "cashback", "возврат", "бонус", "возврат средств"],
        "cashback": ["cashback", "кэшбэк", "возврат", "бонус"],
        "бонус": ["бонус", "bonus", "кэшбэк", "cashback", "награда"],
        "мили": ["мили", "miles", "бонусные мили", "мили карты"],
        "баллы": ["баллы", "points", "бонусные баллы", "баллы программы"],
        
        # Security
        "пароль": ["пароль", "password", "код доступа", "pin"],
        "пин": ["пин", "pin", "пин-код", "персональный идентификационный номер"],
        "пин-код": ["пин-код", "pin code", "pin", "пин"],
        "cvv": ["cvv", "cvc", "код безопасности", "cvv код", "cvc код"],
        "код безопасности": ["код безопасности", "cvv", "cvc", "security code"],
        
        # Additional banking terms
        "овердрафт": ["овердрафт", "overdraft", "кредитный лимит"],
        "аккредитив": ["аккредитив", "letter of credit", "lc"],
        "гарантия": ["гарантия", "guarantee", "банковская гарантия", "банковская гарантия"],
        "инкассо": ["инкассо", "collection", "банковское инкассо"],
        
        # Installments and payments
        "рассрочка": ["рассрочка", "installment", "рассрочка платежа", "рассрочка покупки", "рассрочка без переплаты"],
        "погашение": ["погашение", "repayment", "погашение кредита", "досрочное погашение", "частичное погашение"],
        "задолженность": ["задолженность", "debt", "долг", "задолженность по кредиту", "задолженность по карте"],
        
        # Salary and payroll
        "зарплата": ["зарплата", "salary", "заработная плата", "зарплатный", "зарплатное перечисление"],
        "зарплатный": ["зарплатный", "salary", "зарплатный счет", "зарплатная карта", "зарплатный счет"],
        "зарплатный счет": ["зарплатный счет", "salary account", "зарплатный счёт", "счет для зарплаты"],
        
        # Account types
        "лицевой счет": ["лицевой счет", "лицевой счёт", "personal account", "лицевой счет клиента", "личный счет"],
        "бизнес счет": ["бизнес счет", "business account", "счет для бизнеса", "расчетный счет бизнеса"],
        "кредитный счет": ["кредитный счет", "credit account", "счет кредита", "кредитный счёт"],
        "кредитный счёт": ["кредитный счёт", "credit account", "кредитный счет"],
        
        # Applications and requests
        "заявка": ["заявка", "application", "заявка на кредит", "заявка на карту", "подать заявку", "заявка одобрена"],
        "одобрить": ["одобрить", "approve", "одобрен", "одобрение", "одобрена заявка"],
        "партнер": ["партнер", "partner", "партнер одобрил", "партнерская программа"],
        
        # Contracts and documents
        "договор": ["договор", "contract", "номер договора", "договор по карте", "договор по кредиту", "ипотечный договор"],
        "ипотечный договор": ["ипотечный договор", "mortgage contract", "договор ипотеки", "ипотечный кредит"],
        "справка": ["справка", "certificate", "справка о реквизитах", "справка о счете", "банковская справка"],
        "выписка": ["выписка", "statement", "выписка по счету", "выписка из банка", "банковская выписка"],
        "скан": ["скан", "scan", "скан реквизитов", "скан договора", "отсканировать"],
        
        # Transactions
        "пополнение": ["пополнение", "top-up", "replenishment", "пополнить", "пополнить счет", "пополнение карты"],
        "списание": ["списание", "debit", "withdrawal", "списать", "списание средств", "списание с карты"],
        "зачисление": ["зачисление", "credit", "deposit", "зачислить", "зачисление средств", "зачисление на счет"],
        "поступление": ["поступление", "receipt", "поступление средств", "поступление денег", "поступили деньги"],
        
        # Cards and plastic
        "кредитка": ["кредитка", "credit card", "кредитная карта", "кред карта", "кред"],
        "пластик": ["пластик", "plastic card", "пластиковая карта", "карта пластик"],
        "детская карта": ["детская карта", "child card", "детская", "карта для ребенка"],
        "перевыпуск": ["перевыпуск", "reissue", "перевыпустить карту", "перевыпуск карты"],
        "активация": ["активация", "activation", "активировать", "активация карты", "активировать карту"],
        "срок действия": ["срок действия", "expiry", "expiration", "срок действия карты", "действует до"],
        
        # Notifications and alerts
        "уведомление": ["уведомление", "notification", "alert", "уведомления", "sms уведомление", "push уведомление"],
        "пуш": ["пуш", "push", "push-уведомление", "пуш уведомление", "push notification"],
        "оповещение": ["оповещение", "notification", "alert", "оповещения", "sms оповещение"],
        
        # Personal account and login
        "кабинет": ["кабинет", "cabinet", "личный кабинет", "онлайн кабинет", "кабинет клиента"],
        "личный кабинет": ["личный кабинет", "personal cabinet", "личный кабинет клиента", "онлайн кабинет"],
        
        # Blocking and restrictions
        "блокировать": ["блокировать", "block", "заблокировать", "блокировка", "заблокирован"],
        "заблокирован": ["заблокирован", "blocked", "блокировка", "заблокировать счет", "заблокировать карту"],
        "блокировка": ["блокировка", "blocking", "block", "блокировка карты", "блокировка счета"],
        
        # Payment systems and methods
        "сбп": ["сбп", "sbp", "система быстрых платежей", "быстрые платежи", "перевод по сбп"],
        "стикер": ["стикер", "sticker", "платежный стикер", "альфа стикер", "платежный стикер альфа"],
        "qr": ["qr", "qr код", "qr-code", "кьюар код", "qr code"],
        "кьюар": ["кьюар", "qr", "qr код", "кьюар код"],
        
        # Insurance
        "страховка": ["страховка", "insurance", "страхование", "номер договора страховки", "страховка по кредиту"],
        "страхование": ["страхование", "insurance", "страховка", "банковское страхование", "страхование жизни"],
        
        # Loyalty and bonuses
        "лояльность": ["лояльность", "loyalty", "программа лояльности", "лояльность банка", "бонус лояльности"],
        "маркетплейс": ["маркетплейс", "marketplace", "маркетплейс кэшбек", "кешбек маркетплейс"],
        
        # Investments and additional services
        "дивиденды": ["дивиденды", "dividends", "дивиденды по акциям", "начисление дивидендов"],
        "купоны": ["купоны", "coupons", "купоны облигаций", "начисление купонов"],
        
        # Additional terms
        "привязать": ["привязать", "link", "attach", "привязать карту", "привязать номер"],
        "отвязать": ["отвязать", "unlink", "detach", "отвязать карту", "отвязать номер"],
        "запись": ["запись", "appointment", "записаться", "запись в отделение", "запись в банк", "отменить запись"],
        "сменить": ["сменить", "change", "сменить номер", "сменить пароль", "сменить адрес"],
        "удалить": ["удалить", "delete", "remove", "удалить карту", "удалить счет"],
        "закрыть": ["закрыть", "close", "закрыть счет", "закрыть карту", "закрыть кредит"],
        "тенге": ["тенге", "kzt", "казахский тенге", "обмен тенге", "менять тенге"],
        
        # Operations and history
        "операция": ["операция", "transaction", "operation", "история операций", "последняя операция"],
        "транзакция": ["транзакция", "transaction", "проверить транзакцию", "история транзакций"],
        "история": ["история", "history", "история операций", "история платежей", "история переводов"],
        "оплата": ["оплата", "payment", "оплатить", "проведение оплаты", "прошла оплата"],
        
        # Limits and amounts
        "лимит": ["лимит", "limit", "кредитный лимит", "лимит по карте", "увеличить лимит", "уменьшить лимит"],
        "кредитный лимит": ["кредитный лимит", "credit limit", "лимит по кредитной карте", "лимит кредитки"],
        
        # Cardholder and user
        "картхолдер": ["картхолдер", "cardholder", "владелец карты", "держатель карты"],
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
    
    # Extract keywords first
    keywords = extract_keywords(enhanced)
    
    # Add synonyms for key banking terms with improved matching
    query_lower = enhanced.lower()
    added_synonyms = set()
    added_phrases = set()
    
    # Priority 1: Exact keyword matches (add more synonyms for important terms)
    for key, synonyms in banking_synonyms.items():
        if key in query_lower:
            # Add more synonyms for important banking terms (up to 5 for critical terms)
            max_synonyms = 5 if key in {"счет", "счёт", "карта", "кредит", "вклад", "реквизиты", "номер"} else 3
            for synonym in synonyms[:max_synonyms]:
                if synonym not in query_lower and synonym not in added_synonyms:
                    enhanced += f" {synonym}"
                    added_synonyms.add(synonym)
        
        # Also check for partial matches in multi-word queries
        for keyword in keywords:
            if key in keyword or keyword in key:
                if len(key) >= 3:  # Only for meaningful matches
                    for synonym in synonyms[:2]:
                        if synonym not in query_lower and synonym not in added_synonyms:
                            enhanced += f" {synonym}"
                            added_synonyms.add(synonym)
                            break
    
    # Priority 2: Multi-word phrase matching (e.g., "номер счета" -> "account number")
    banking_phrases = {
        "номер счета": ["номер счета", "номер счёта", "account number", "счет номер", "счёт номер"],
        "номер карты": ["номер карты", "card number", "карта номер", "card num"],
        "банковские реквизиты": ["банковские реквизиты", "banking details", "bank details", "банковские детали"],
        "процентная ставка": ["процентная ставка", "interest rate", "ставка процента", "процент"],
        "годовая ставка": ["годовая ставка", "annual rate", "годовой процент", "annual interest"],
        "мобильный банк": ["мобильный банк", "mobile bank", "mobile banking", "мобильное приложение"],
        "интернет-банк": ["интернет-банк", "internet bank", "online bank", "интернет банк"],
        "кредитная карта": ["кредитная карта", "credit card", "кредитка", "cc"],
        "дебетовая карта": ["дебетовая карта", "debit card", "дебетка", "dc"],
        "банковское отделение": ["банковское отделение", "bank branch", "банк отделение", "office branch"],
        "зарплатный счет": ["зарплатный счет", "salary account", "зарплатный счёт", "счет для зарплаты"],
        "кредитный счет": ["кредитный счет", "credit account", "кредитный счёт"],
        "лицевой счет": ["лицевой счет", "personal account", "лицевой счёт"],
        "рассрочка платежа": ["рассрочка", "installment", "рассрочка без переплаты"],
        "номер договора": ["номер договора", "contract number", "договор номер"],
        "личный кабинет": ["личный кабинет", "personal cabinet", "кабинет клиента"],
        "платежный стикер": ["платежный стикер", "payment sticker", "альфа стикер", "стикер"],
        "система быстрых платежей": ["сбп", "sbp", "система быстрых платежей"],
        "push уведомление": ["push уведомление", "push notification", "пуш", "push-уведомление"],
        "досрочное погашение": ["досрочное погашение", "early repayment", "погасить досрочно"],
        "история операций": ["история операций", "transaction history", "история платежей"],
        "кредитный лимит": ["кредитный лимит", "credit limit", "лимит по карте"],
        "заблокировать карту": ["заблокировать карту", "block card", "блокировка карты"],
        "перевыпуск карты": ["перевыпуск карты", "reissue card", "перевыпустить карту"],
        "активация карты": ["активация карты", "card activation", "активировать карту"],
    }
    
    for phrase, variants in banking_phrases.items():
        phrase_lower = phrase.lower()
        if phrase_lower in query_lower:
            for variant in variants[:2]:
                if variant not in query_lower and variant not in added_phrases:
                    enhanced += f" {variant}"
                    added_phrases.add(variant)
    
    # Add query rewriting patterns
    for pattern, rewrites in query_rewrites.items():
        if pattern in query_lower:
            for rewrite in rewrites[:2]:  # Add first 2 rewrites
                if rewrite not in query_lower and rewrite not in added_synonyms:
                    enhanced += f" {rewrite}"
                    added_synonyms.add(rewrite)
    
    # Priority 3: Add related terms based on keyword analysis
    keyword_relations = {
        "счет": ["баланс", "выписка", "реквизиты", "пополнение", "списание"],
        "карта": ["лимит", "баланс", "pin", "cvv", "активация", "перевыпуск"],
        "кредит": ["процент", "ставка", "лимит", "погашение", "задолженность", "договор"],
        "вклад": ["процент", "ставка", "накопления", "сбережения"],
        "перевод": ["комиссия", "реквизиты", "платеж", "сбп", "история"],
        "платеж": ["комиссия", "перевод", "транзакция", "операция"],
        "номер": ["счет", "карта", "реквизиты", "договор"],
        "реквизиты": ["бик", "инн", "счет", "банк", "скан", "справка"],
        "рассрочка": ["погашение", "досрочно", "закрыть", "счет"],
        "заявка": ["одобрить", "одобрена", "партнер"],
        "уведомление": ["смс", "пуш", "sms", "push"],
        "договор": ["номер", "ипотека", "страховка", "карта"],
        "пополнение": ["зачисление", "поступление", "история"],
        "списание": ["операция", "история", "транзакция"],
        "зарплата": ["зарплатный счет", "зачисление", "поступление"],
    }
    
    for keyword in keywords:
        if keyword in keyword_relations:
            for related_term in keyword_relations[keyword]:
                if related_term not in query_lower and related_term not in added_synonyms:
                    enhanced += f" {related_term}"
                    added_synonyms.add(related_term)
                    break  # Add one related term per keyword to avoid over-expansion
    
    # Extract numbers from query and add variations (improved matching)
    numbers = re.findall(r'\d+[.,]?\d*', query)
    for num in numbers:
        # Add variations: with/without spaces, with/without decimal
        if '.' in num or ',' in num:
            # Add integer version
            int_version = re.sub(r'[.,]\d+', '', num)
            if int_version and int_version not in enhanced.lower():
                enhanced += f" {int_version}"
            # Also add normalized version (replace comma with dot)
            normalized = num.replace(',', '.')
            if normalized != num and normalized not in enhanced.lower():
                enhanced += f" {normalized}"
        else:
            # Add decimal versions for better matching
            if num not in enhanced.replace('.', ' ').replace(',', ' '):
                enhanced += f" {num}.0 {num},0"
                # Also add space-separated version for large numbers
                if len(num) > 3:
                    spaced = ' '.join(num[i:i+3] for i in range(0, len(num), 3))
                    enhanced += f" {spaced}"
    
    # Add common banking query patterns with improved context detection
    query_lower_final = enhanced.lower()
    if any(word in query_lower_final for word in ["где", "как", "какой", "что", "как узнать", "как найти"]):
        # Add "информация" and "помощь" for informational queries
        if "информация" not in query_lower_final:
            enhanced += " информация"
        if "условия" not in query_lower_final and any(w in query_lower_final for w in ["кредит", "вклад", "карта", "депозит"]):
            enhanced += " условия"
        if any(w in query_lower_final for w in ["оформить", "получить", "заказать"]) and "инструкция" not in query_lower_final:
            enhanced += " инструкция"
    
    # Boost important keywords by repetition (subtle boost for BM25)
    important_keywords = ["реквизиты", "номер", "счет", "карта", "баланс"]
    for keyword in important_keywords:
        if keyword in query_lower_final and keyword not in added_synonyms:
            # Don't add again, but ensure it's present (already handled above)
            pass
    
    # Remove extra spaces and normalize
    enhanced = re.sub(r'\s+', ' ', enhanced).strip()
    
    # Limit to reasonable length (keep original + ~200 chars expansion for better quality)
    max_expansion = 200
    if len(enhanced) > len(query) + max_expansion and query:
        # Keep original query + prioritize important expansions
        try:
            words = enhanced.split() if enhanced else []
            original_words = query.split() if query else []
            if not words or not original_words:
                # If split fails, just truncate
                enhanced = enhanced[:len(query) + max_expansion]
            else:
                max_expansion_words = 40  # Increased from 30
                # Prioritize: keep original + abbreviations expansion + first keywords + synonyms
                enhanced_parts = []
                enhanced_parts.extend(original_words)
                
                # Add expansion words, prioritizing non-stop-words
                expansion_words = words[len(original_words):]
                added_count = 0
                for word in expansion_words:
                    if added_count >= max_expansion_words:
                        break
                    if word.lower() not in {"и", "в", "на", "с", "по", "для"}:
                        enhanced_parts.append(word)
                        added_count += 1
                
                enhanced = " ".join(enhanced_parts)
        except (AttributeError, IndexError, TypeError):
            # If anything fails, just truncate safely
            enhanced = enhanced[:len(query) + max_expansion] if query else enhanced
    
    # Final normalization
    enhanced = re.sub(r'\s+', ' ', enhanced).strip()
    
    return enhanced

