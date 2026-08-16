import requests
import json
import pandas as pd
from json_repair import repair_json
import pyarabic.araby as araby

def clean_arabic_text(text):
    text = str(text)
    text = araby.strip_tashkeel(text)
    text = araby.normalize_alef(text)
    return text

def load_pragmagap_data(url="https://raw.githubusercontent.com/NoorBayan/Burhan/main/corpus/metaphors_data.json"):
    response = requests.get(url)
    fixed_json_string = repair_json(response.text)
    data = json.loads(fixed_json_string)

    records = []

    for item in data:
        ayah = item.get('metadata', {}).get('ayah_text_uthmani', '')
        similes = item.get('rhetorical_analysis', {}).get('similes', []) # يمثل الاستعارات هنا
        
        if not similes: continue
        
        for metaphor in similes:
            segment = metaphor.get('simile_identity', {}).get('segment_text', '')
            
            # استخراج البنية النحوية
            syntax_val = metaphor.get('syntactic_structure', {}).get('grammatical_structure', '')
            
            # استخراج الفعل الكلامي (نأخذ الأول إن وجد)
            functions = metaphor.get('functions', [])
            speech_act_val = functions[0].get('speech_act', '') if functions else ''

            if syntax_val and speech_act_val and ayah:
                # دمج الآية مع المقطع المستهدف
                combined_text = f"{ayah} [SEP] {segment}" if segment else ayah
                
                # Binarization Task A: Syntax (Verbal/Dynamic vs Nominal/Static)
                if syntax_val in ['verbal_structure', 'adverbial_structure']:
                    syntax_label = 0 # Dynamic / Action
                else:
                    syntax_label = 1 # Static / Nominal / Descriptive
                    
                # Binarization Task B: Pragmatics (Assertive vs Affective/Directive)
                if speech_act_val == 'ASSERTIVE':
                    pragma_label = 0 # Informative / Assertive
                else:
                    pragma_label = 1 # Affective / Directive / Expressive

                records.append({
                    'text': combined_text, 
                    'syntax_raw': syntax_val,
                    'pragma_raw': speech_act_val,
                    'label_syntax': syntax_label,
                    'label_pragma': pragma_label
                })

    df = pd.DataFrame(records)
    df['clean_text'] = df['text'].apply(clean_arabic_text)
    
    # تعريف أسماء الفئات للتحليل لاحقاً
    label_encoders = {
        'syntax': {0: "Verbal/Dynamic", 1: "Nominal/Static"},
        'pragma': {0: "Assertive/Informative", 1: "Affective/Action"}
    }
    
    return df, label_encoders
