"""
Healthcare-specific utilities for the Q&A Assistant.
Contains medical terminology, drug information processing, and healthcare domain helpers.
"""

import re
from typing import Dict, List, Set, Tuple, Any
from dataclasses import dataclass
from datetime import datetime

@dataclass
class MedicalEntity:
    """Represents a medical entity (drug, condition, procedure, etc.)."""
    name: str
    category: str
    synonyms: List[str]
    description: str = ""

class HealthcareTerminologyProcessor:
    """
    Processes and enhances healthcare-specific terminology.
    Handles medical abbreviations, drug names, and clinical terms.
    """
    
    def __init__(self):
        """Initialize with medical terminology databases."""
        self._load_medical_dictionaries()
    
    def _load_medical_dictionaries(self):
        """Load medical dictionaries and terminologies."""
        
        # Common medical abbreviations
        self.medical_abbreviations = {
            # Dosing and frequency
            'q.d.': 'once daily',
            'qd': 'once daily',
            'b.i.d.': 'twice daily',
            'bid': 'twice daily',
            't.i.d.': 'three times daily',
            'tid': 'three times daily',
            'q.i.d.': 'four times daily',
            'qid': 'four times daily',
            'q4h': 'every 4 hours',
            'q6h': 'every 6 hours',
            'q8h': 'every 8 hours',
            'q12h': 'every 12 hours',
            'prn': 'as needed',
            'p.r.n.': 'as needed',
            'ac': 'before meals',
            'pc': 'after meals',
            'hs': 'at bedtime',
            
            # Routes of administration
            'po': 'oral',
            'p.o.': 'oral',
            'iv': 'intravenous',
            'i.v.': 'intravenous',
            'im': 'intramuscular',
            'i.m.': 'intramuscular',
            'sc': 'subcutaneous',
            's.c.': 'subcutaneous',
            'sl': 'sublingual',
            'top': 'topical',
            
            # Units
            'mg': 'milligrams',
            'mcg': 'micrograms',
            'g': 'grams',
            'kg': 'kilograms',
            'ml': 'milliliters',
            'l': 'liters',
            'cc': 'cubic centimeters',
            'iu': 'international units',
            'meq': 'milliequivalents',
            'mmol': 'millimoles',
            
            # Clinical terms
            'hx': 'history',
            'dx': 'diagnosis',
            'tx': 'treatment',
            'rx': 'prescription',
            'sx': 'symptoms',
            'fx': 'fracture',
            'pt': 'patient',
            'pts': 'patients',
            'htn': 'hypertension',
            'dm': 'diabetes mellitus',
            'cad': 'coronary artery disease',
            'chf': 'congestive heart failure',
            'copd': 'chronic obstructive pulmonary disease',
            'uti': 'urinary tract infection',
            'uri': 'upper respiratory infection',
            'mi': 'myocardial infarction',
            'cvd': 'cardiovascular disease',
            'cva': 'cerebrovascular accident',
            'tia': 'transient ischemic attack',
            
            # Laboratory values
            'wbc': 'white blood cell count',
            'rbc': 'red blood cell count',
            'hgb': 'hemoglobin',
            'hct': 'hematocrit',
            'plt': 'platelet count',
            'bun': 'blood urea nitrogen',
            'cr': 'creatinine',
            'glucose': 'blood glucose',
            'hba1c': 'hemoglobin a1c',
            'ldl': 'low-density lipoprotein',
            'hdl': 'high-density lipoprotein',
            'tsh': 'thyroid stimulating hormone',
            'inr': 'international normalized ratio',
            
            # Vital signs
            'bp': 'blood pressure',
            'hr': 'heart rate',
            'rr': 'respiratory rate',
            'temp': 'temperature',
            'o2sat': 'oxygen saturation',
            'spo2': 'pulse oximetry'
        }
        
        # Drug categories and common medications
        self.drug_categories = {
            'antibiotics': [
                'amoxicillin', 'azithromycin', 'ciprofloxacin', 'clindamycin',
                'doxycycline', 'erythromycin', 'levofloxacin', 'metronidazole',
                'penicillin', 'trimethoprim-sulfamethoxazole', 'vancomycin'
            ],
            'antihypertensives': [
                'lisinopril', 'amlodipine', 'metoprolol', 'losartan',
                'hydrochlorothiazide', 'atenolol', 'carvedilol', 'enalapril',
                'valsartan', 'chlorthalidone'
            ],
            'diabetes_medications': [
                'metformin', 'insulin', 'glipizide', 'sitagliptin',
                'empagliflozin', 'liraglutide', 'pioglitazone', 'glyburide'
            ],
            'pain_medications': [
                'acetaminophen', 'ibuprofen', 'naproxen', 'aspirin',
                'tramadol', 'morphine', 'oxycodone', 'hydrocodone',
                'celecoxib', 'diclofenac'
            ],
            'cardiac_medications': [
                'atorvastatin', 'simvastatin', 'clopidogrel', 'warfarin',
                'digoxin', 'furosemide', 'spironolactone', 'isosorbide'
            ],
            'psychiatric_medications': [
                'sertraline', 'fluoxetine', 'escitalopram', 'paroxetine',
                'venlafaxine', 'bupropion', 'trazodone', 'mirtazapine',
                'risperidone', 'quetiapine', 'aripiprazole'
            ]
        }
        
        # Medical specialties
        self.medical_specialties = {
            'cardiology': ['heart', 'cardiac', 'cardiovascular', 'coronary'],
            'endocrinology': ['diabetes', 'thyroid', 'hormone', 'endocrine'],
            'gastroenterology': ['digestive', 'stomach', 'intestinal', 'liver'],
            'nephrology': ['kidney', 'renal', 'dialysis', 'urinary'],
            'neurology': ['brain', 'neurological', 'seizure', 'stroke'],
            'pulmonology': ['lung', 'respiratory', 'breathing', 'asthma'],
            'rheumatology': ['arthritis', 'joint', 'autoimmune', 'inflammation'],
            'infectious_disease': ['infection', 'bacterial', 'viral', 'antimicrobial']
        }
        
        # Drug interaction classes
        self.interaction_classes = {
            'major': 'Avoid combination - risk of serious adverse effects',
            'moderate': 'Monitor closely - may require dose adjustment',
            'minor': 'Monitor - unlikely to cause significant problems'
        }
    
    def expand_abbreviations(self, text: str) -> str:
        """
        Expand medical abbreviations in text.
        
        Args:
            text (str): Input text with abbreviations
            
        Returns:
            str: Text with expanded abbreviations
        """
        expanded_text = text.lower()
        
        # Sort by length (longest first) to avoid partial replacements
        sorted_abbrevs = sorted(self.medical_abbreviations.items(), 
                               key=lambda x: len(x[0]), reverse=True)
        
        for abbrev, expansion in sorted_abbrevs:
            # Use word boundaries to avoid partial matches
            pattern = r'\b' + re.escape(abbrev) + r'\b'
            expanded_text = re.sub(pattern, expansion, expanded_text, flags=re.IGNORECASE)
        
        return expanded_text
    
    def identify_drug_mentions(self, text: str) -> List[Tuple[str, str]]:
        """
        Identify drug mentions in text and their categories.
        
        Args:
            text (str): Input text
            
        Returns:
            List[Tuple[str, str]]: List of (drug_name, category) tuples
        """
        found_drugs = []
        text_lower = text.lower()
        
        for category, drugs in self.drug_categories.items():
            for drug in drugs:
                if drug.lower() in text_lower:
                    found_drugs.append((drug, category))
        
        return found_drugs
    
    def identify_medical_specialty(self, text: str) -> List[str]:
        """
        Identify relevant medical specialties based on text content.
        
        Args:
            text (str): Input text
            
        Returns:
            List[str]: List of relevant medical specialties
        """
        relevant_specialties = []
        text_lower = text.lower()
        
        for specialty, keywords in self.medical_specialties.items():
            if any(keyword in text_lower for keyword in keywords):
                relevant_specialties.append(specialty)
        
        return relevant_specialties
    
    def extract_dosage_information(self, text: str) -> List[Dict[str, str]]:
        """
        Extract dosage information from text.
        
        Args:
            text (str): Input text
            
        Returns:
            List[Dict[str, str]]: List of dosage information dictionaries
        """
        dosage_patterns = [
            # Standard dosage patterns
            r'(\d+(?:\.\d+)?)\s*(mg|mcg|g|ml|units?|iu)\s*(?:(?:every|q)\s*(\d+)\s*(?:hours?|hrs?|h)|(?:once|twice|three times|four times)\s*(?:daily|a day)|(?:bid|tid|qid|qd))',
            # Range dosages
            r'(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)\s*(mg|mcg|g|ml|units?|iu)',
            # PRN dosages
            r'(\d+(?:\.\d+)?)\s*(mg|mcg|g|ml|units?|iu)\s*(?:as needed|prn|p\.r\.n\.)'
        ]
        
        dosages = []
        for pattern in dosage_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                dosage_info = {
                    'full_match': match.group(0),
                    'dose': match.group(1),
                    'unit': match.group(2) if len(match.groups()) >= 2 else '',
                    'frequency': match.group(3) if len(match.groups()) >= 3 else ''
                }
                dosages.append(dosage_info)
        
        return dosages
    
    def extract_contraindications(self, text: str) -> List[str]:
        """
        Extract contraindication information from text.
        
        Args:
            text (str): Input text
            
        Returns:
            List[str]: List of contraindications
        """
        contraindication_patterns = [
            r'contraindicated?\s+(?:in|for|with)\s+([^.]+)',
            r'should\s+not\s+be\s+used\s+(?:in|for|with)\s+([^.]+)',
            r'avoid\s+(?:in|for|with)\s+([^.]+)',
            r'do\s+not\s+use\s+(?:in|for|with)\s+([^.]+)'
        ]
        
        contraindications = []
        for pattern in contraindication_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                contraindication = match.group(1).strip()
                contraindications.append(contraindication)
        
        return contraindications
    
    def extract_side_effects(self, text: str) -> List[str]:
        """
        Extract side effects from text.
        
        Args:
            text (str): Input text
            
        Returns:
            List[str]: List of side effects
        """
        side_effect_patterns = [
            r'side\s+effects?\s+(?:include|may include|are)?\s*:?\s*([^.]+)',
            r'adverse\s+(?:effects?|reactions?)\s+(?:include|may include|are)?\s*:?\s*([^.]+)',
            r'may\s+cause\s+([^.]+)',
            r'common\s+(?:side\s+effects?|adverse\s+reactions?)\s+(?:include|are)?\s*:?\s*([^.]+)'
        ]
        
        side_effects = []
        for pattern in side_effect_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                effects_text = match.group(1).strip()
                # Split by common delimiters
                effects = re.split(r'[,;]|\s+and\s+|\s+or\s+', effects_text)
                side_effects.extend([effect.strip() for effect in effects if effect.strip()])
        
        return side_effects

class DrugInteractionChecker:
    """
    Simple drug interaction checker for common medications.
    """
    
    def __init__(self):
        """Initialize with basic drug interaction data."""
        self._load_interaction_data()
    
    def _load_interaction_data(self):
        """Load basic drug interaction data."""
        # This is a simplified interaction database
        # In a real system, this would be much more comprehensive
        self.interactions = {
            ('warfarin', 'aspirin'): {
                'severity': 'major',
                'description': 'Increased risk of bleeding',
                'management': 'Monitor INR closely, consider alternatives'
            },
            ('lisinopril', 'potassium'): {
                'severity': 'moderate',
                'description': 'Risk of hyperkalemia',
                'management': 'Monitor potassium levels'
            },
            ('metformin', 'contrast_dye'): {
                'severity': 'major',
                'description': 'Risk of lactic acidosis',
                'management': 'Hold metformin before contrast procedures'
            },
            ('digoxin', 'furosemide'): {
                'severity': 'moderate',
                'description': 'Increased digoxin toxicity risk',
                'management': 'Monitor digoxin levels and potassium'
            }
        }
    
    def check_interactions(self, drug_list: List[str]) -> List[Dict[str, Any]]:
        """
        Check for drug interactions in a list of medications.
        
        Args:
            drug_list (List[str]): List of drug names
            
        Returns:
            List[Dict[str, Any]]: List of interaction information
        """
        found_interactions = []
        
        # Normalize drug names
        normalized_drugs = [drug.lower().strip() for drug in drug_list]
        
        # Check all combinations
        for i, drug1 in enumerate(normalized_drugs):
            for drug2 in normalized_drugs[i+1:]:
                # Check both orders
                interaction_key1 = (drug1, drug2)
                interaction_key2 = (drug2, drug1)
                
                interaction = None
                if interaction_key1 in self.interactions:
                    interaction = self.interactions[interaction_key1]
                elif interaction_key2 in self.interactions:
                    interaction = self.interactions[interaction_key2]
                
                if interaction:
                    interaction_info = {
                        'drug1': drug1,
                        'drug2': drug2,
                        'severity': interaction['severity'],
                        'description': interaction['description'],
                        'management': interaction['management']
                    }
                    found_interactions.append(interaction_info)
        
        return found_interactions

class HealthcareSafetyChecker:
    """
    Checks for potential safety issues in healthcare queries and responses.
    """
    
    def __init__(self):
        """Initialize safety checker."""
        self.safety_keywords = {
            'high_risk': [
                'emergency', 'urgent', 'cardiac arrest', 'stroke', 'seizure',
                'severe allergic reaction', 'anaphylaxis', 'overdose',
                'suicidal', 'chest pain', 'difficulty breathing'
            ],
            'dosage_concerns': [
                'maximum dose', 'overdose', 'toxic dose', 'lethal dose',
                'double dose', 'missed dose'
            ],
            'pregnancy_concerns': [
                'pregnancy', 'pregnant', 'breastfeeding', 'nursing',
                'teratogenic', 'fetal'
            ]
        }
    
    def check_safety_concerns(self, text: str) -> Dict[str, List[str]]:
        """
        Check for safety concerns in text.
        
        Args:
            text (str): Input text to check
            
        Returns:
            Dict[str, List[str]]: Dictionary of safety concerns by category
        """
        concerns = {}
        text_lower = text.lower()
        
        for category, keywords in self.safety_keywords.items():
            found_keywords = []
            for keyword in keywords:
                if keyword in text_lower:
                    found_keywords.append(keyword)
            
            if found_keywords:
                concerns[category] = found_keywords
        
        return concerns
    
    def generate_safety_warning(self, concerns: Dict[str, List[str]]) -> str:
        """
        Generate appropriate safety warnings based on concerns.
        
        Args:
            concerns (Dict[str, List[str]]): Safety concerns
            
        Returns:
            str: Safety warning message
        """
        warnings = []
        
        if 'high_risk' in concerns:
            warnings.append(
                "⚠️ HIGH PRIORITY: This query involves potentially urgent medical situations. "
                "For immediate medical emergencies, contact emergency services immediately."
            )
        
        if 'dosage_concerns' in concerns:
            warnings.append(
                "⚠️ DOSAGE WARNING: This query involves medication dosing. "
                "Always verify dosing information with current prescribing information "
                "and consider patient-specific factors."
            )
        
        if 'pregnancy_concerns' in concerns:
            warnings.append(
                "⚠️ PREGNANCY/NURSING WARNING: This query involves pregnancy or nursing considerations. "
                "Always consult current pregnancy/lactation guidelines and consider "
                "risk-benefit ratios."
            )
        
        if warnings:
            return "\n\n".join(warnings)
        
        return ""

# Utility functions for healthcare data processing
def extract_vital_signs(text: str) -> Dict[str, str]:
    """
    Extract vital signs from text.
    
    Args:
        text (str): Input text
        
    Returns:
        Dict[str, str]: Dictionary of vital signs
    """
    vital_patterns = {
        'blood_pressure': r'(?:bp|blood pressure)\s*:?\s*(\d{2,3})/(\d{2,3})',
        'heart_rate': r'(?:hr|heart rate|pulse)\s*:?\s*(\d{2,3})',
        'temperature': r'(?:temp|temperature)\s*:?\s*(\d{2,3}(?:\.\d)?)\s*°?([cf])?',
        'respiratory_rate': r'(?:rr|respiratory rate|respiration)\s*:?\s*(\d{1,2})',
        'oxygen_saturation': r'(?:o2|spo2|oxygen saturation)\s*:?\s*(\d{2,3})%?'
    }
    
    vitals = {}
    for vital, pattern in vital_patterns.items():
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            vitals[vital] = match.group(0)
    
    return vitals

def standardize_medical_units(text: str) -> str:
    """
    Standardize medical units in text.
    
    Args:
        text (str): Input text
        
    Returns:
        str: Text with standardized units
    """
    unit_conversions = {
        'mcg': 'micrograms',
        'μg': 'micrograms',
        'cc': 'mL',
        'gm': 'g',
        'gms': 'g',
        'kgs': 'kg',
        'lbs': 'pounds'
    }
    
    standardized_text = text
    for old_unit, new_unit in unit_conversions.items():
        pattern = r'\b' + re.escape(old_unit) + r'\b'
        standardized_text = re.sub(pattern, new_unit, standardized_text, flags=re.IGNORECASE)
    
    return standardized_text