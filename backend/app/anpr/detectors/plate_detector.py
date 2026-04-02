"""Indian Plate Detector - Validates Indian license plate format"""
import re


class IndianPlateDetector:
    """OCR and validation for Indian license plates."""
    
    VALID_STATE_CODES = {
        'AP', 'AR', 'AS', 'BR', 'CG', 'CH', 'CT', 'DD', 'DL', 'DN',
        'GA', 'GJ', 'HR', 'HP', 'JK', 'JH', 'KA', 'KL', 'LD', 'MP',
        'MH', 'MN', 'ML', 'MZ', 'NL', 'OD', 'PB', 'PY', 'RJ', 'SK',
        'TG', 'TS', 'TR', 'TN', 'UP', 'UT', 'WB',
    }
    
    OCR_CHAR_CORRECTIONS = {
        'O': '0', 'I': '1', 'B': '8', 'S': '5', 'Z': '7', 'G': '6', 'A': '4',
    }
    
    REVERSE_CHAR_CORRECTIONS = {
        '0': 'O', '1': 'I', '8': 'B', '5': 'S', '7': 'Z', '6': 'G',
    }
    
    REVERSE_CHAR_ALTERNATIVES = {
        '1': 'T', '2': 'Z',
    }
    
    LETTER_CONFUSIONS = {}
    DIGIT_CORRECTIONS = {}
    DIGIT_ALTERNATIVES = {
        'A': ['4'],
    }
    
    @staticmethod
    def normalize_ocr_text(raw_text):
        """Normalize OCR output: uppercase, remove spaces, standardize hyphens."""
        if not raw_text:
            return ""
        
        text = raw_text.strip().upper()
        text = re.sub(r'\s+', '', text)
        text = re.sub(r'[–—−_•·•]', '-', text)
        text = re.sub(r'[^A-Z0-9\-]', '', text)
        
        return text
    
    @staticmethod
    def apply_ocr_corrections(text, format_type='standard'):
        """Correct common OCR character misclassifications based on plate position."""
        if not text:
            return ""
        
        text = list(text.replace('-', ''))
        
        # Plate position constraints (type: 'alpha' for letters, 'numeric' for digits)
        if format_type == 'standard':
            corrections = [
                (0, 1, 'alpha'),    # State code
                (2, 3, 'numeric'),  # District code
                (4, 5, 'alpha'),    # Category
                (6, 9, 'numeric'),  # Serial
            ]
        elif format_type == 'bharat':
            corrections = [
                (0, 1, 'numeric'),  # Year
                (2, 3, 'alpha'),    # BH
                (4, 7, 'numeric'),  # Serial
                (8, None, 'alpha'), # Series
            ]
        else:
            return ''.join(text)
        
        for start, end, pos_type in corrections:
            if end is None:
                end = len(text) - 1
            
            for i in range(start, min(end + 1, len(text))):
                char = text[i]
                
                if pos_type == 'numeric':
                    if char in IndianPlateDetector.OCR_CHAR_CORRECTIONS:
                        text[i] = IndianPlateDetector.OCR_CHAR_CORRECTIONS[char]
                    elif char in IndianPlateDetector.DIGIT_CORRECTIONS:
                        text[i] = IndianPlateDetector.DIGIT_CORRECTIONS[char]
                elif pos_type == 'alpha':
                    if char in IndianPlateDetector.REVERSE_CHAR_CORRECTIONS:
                        text[i] = IndianPlateDetector.REVERSE_CHAR_CORRECTIONS[char]
                    elif char in IndianPlateDetector.LETTER_CONFUSIONS:
                        text[i] = IndianPlateDetector.LETTER_CONFUSIONS[char]
        
        return ''.join(text)
    
    @staticmethod
    def apply_alternative_ocr_corrections(text, format_type='standard'):
        """Apply alternative character mappings for difficult OCR cases."""
        if not text:
            return ""
        
        text = list(text.replace('-', ''))
        
        if format_type == 'standard':
            corrections = [(0, 1, 'alpha'), (2, 3, 'numeric'), (4, 5, 'alpha'), (6, 9, 'numeric')]
        elif format_type == 'bharat':
            corrections = [(0, 1, 'numeric'), (2, 3, 'alpha'), (4, 7, 'numeric'), (8, None, 'alpha')]
        else:
            return ''.join(text)
        
        for start, end, pos_type in corrections:
            if end is None:
                end = len(text) - 1
            
            for i in range(start, min(end + 1, len(text))):
                char = text[i]
                
                if pos_type == 'alpha':
                    if char in IndianPlateDetector.REVERSE_CHAR_ALTERNATIVES:
                        text[i] = IndianPlateDetector.REVERSE_CHAR_ALTERNATIVES[char]
                elif pos_type == 'numeric':
                    if char in IndianPlateDetector.DIGIT_ALTERNATIVES:
                        alternatives = IndianPlateDetector.DIGIT_ALTERNATIVES[char]
                        if alternatives:
                            text[i] = alternatives[0]
        
        return ''.join(text)
    
    @staticmethod
    def validate_standard_format(normalized_text):
        """Validate standard state registration format (SS-NN-AA-NNNN)."""
        if not normalized_text or len(normalized_text) < 9:
            return False, "", 0.0
        
        pattern_full = re.match(r'^([A-Z]{2})-([0-9]{2})-([A-Z]{1,2})-([0-9]{4})$', normalized_text)
        pattern_variant = re.match(r'^([A-Z]{2})-([0-9]{2})-([A-Z])-([0-9]{4})$', normalized_text)
        text_clean = normalized_text.replace('-', '')
        pattern_flexible = re.match(r'^([A-Z]{2})([0-9]{2})([A-Z]{1,2})([0-9]{4})$', text_clean)
        
        pattern = pattern_full or pattern_variant or pattern_flexible
        if not pattern:
            return False, "", 0.0
        
        state, district, category, serial = pattern.groups()
        
        if state.upper() not in IndianPlateDetector.VALID_STATE_CODES:
            return False, "", 0.0
        
        district_num = int(district)
        if district_num < 1 or district_num > 99:
            return False, "", 0.0
        
        if not category.isalpha() or len(category) > 2:
            return False, "", 0.0
        
        if not serial.isdigit() or len(serial) != 4:
            return False, "", 0.0
        
        normalized = f"{state}-{district}-{category}-{serial}"
        return True, normalized, 0.95
    
    @staticmethod
    def validate_bharat_format(normalized_text):
        """Validate Bharat (BH) series format (YY-BH-NNNN-A/AA)."""
        if not normalized_text or len(normalized_text) < 9:
            return False, "", 0.0
        
        pattern_single = re.match(r'^([0-9]{2})-BH-([0-9]{4})-([A-Z])$', normalized_text)
        pattern_double = re.match(r'^([0-9]{2})-BH-([0-9]{4})-([A-Z]{2})$', normalized_text)
        text_clean = normalized_text.replace('-', '')
        pattern_flexible = re.match(r'^([0-9]{2})BH([0-9]{4})([A-Z]{1,2})$', text_clean)
        
        pattern = pattern_single or pattern_double or pattern_flexible
        if not pattern:
            return False, "", 0.0
        
        year, serial, series = pattern.groups()
        
        year_num = int(year)
        if year_num < 8 or year_num > 29:
            return False, "", 0.0
        
        if not serial.isdigit() or len(serial) != 4:
            return False, "", 0.0
        
        if not series.isalpha() or len(series) > 2:
            return False, "", 0.0
        
        normalized = f"{year}-BH-{serial}-{series}"
        return True, normalized, 0.90
    
    @staticmethod
    def is_valid_indian_plate(plate_text):
        """Validate Indian license plate with OCR corrections (standard and Bharat formats)."""
        if not plate_text or len(plate_text) < 8:
            return False, "", 0.0
        
        normalized = IndianPlateDetector.normalize_ocr_text(plate_text)
        if not normalized:
            return False, "", 0.0
        
        is_valid, formatted, conf = IndianPlateDetector.validate_standard_format(normalized)
        if is_valid:
            return True, formatted, conf
        
        is_valid, formatted, conf = IndianPlateDetector.validate_bharat_format(normalized)
        if is_valid:
            return True, formatted, conf
        
        corrected_standard = IndianPlateDetector.apply_ocr_corrections(normalized, format_type='standard')
        is_valid, formatted, conf = IndianPlateDetector.validate_standard_format(corrected_standard)
        if is_valid:
            return True, formatted, conf
        
        corrected_bharat = IndianPlateDetector.apply_ocr_corrections(normalized, format_type='bharat')
        is_valid, formatted, conf = IndianPlateDetector.validate_bharat_format(corrected_bharat)
        if is_valid:
            return True, formatted, conf
        
        corrected_alt_standard = IndianPlateDetector.apply_alternative_ocr_corrections(normalized, format_type='standard')
        if corrected_alt_standard != normalized:
            is_valid, formatted, conf = IndianPlateDetector.validate_standard_format(corrected_alt_standard)
            if is_valid:
                return True, formatted, conf
        
        corrected_alt_bharat = IndianPlateDetector.apply_alternative_ocr_corrections(normalized, format_type='bharat')
        if corrected_alt_bharat != normalized:
            is_valid, formatted, conf = IndianPlateDetector.validate_bharat_format(corrected_alt_bharat)
            if is_valid:
                return True, formatted, conf
        
        return False, "", 0.0
