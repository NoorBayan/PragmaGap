from scipy.stats import wilcoxon
import numpy as np

def run_gap_significance_test(true_syntax, preds_syntax, true_pragma, preds_pragma):
    """
    يقارن ما إذا كانت أخطاء النموذج في مهمة النحو تختلف إحصائياً عن أخطائه في مهمة التداوليات.
    """
    # 0 يعني إجابة صحيحة، 1 يعني إجابة خاطئة
    errors_syntax = (true_syntax != preds_syntax).astype(int)
    errors_pragma = (true_pragma != preds_pragma).astype(int)
    
    if np.array_equal(errors_syntax, errors_pragma):
        return 1.0 
        
    stat, p_value = wilcoxon(errors_syntax, errors_pragma)
    return p_value
