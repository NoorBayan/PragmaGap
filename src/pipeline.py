import numpy as np
import time
import torch
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, set_seed
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.utils.class_weight import compute_class_weight
from .custom_trainer import WeightedTrainer
from .metrics import compute_metrics

def run_5fold_cv_dual_task(model_name, df, target_task='syntax', k_folds=5, seed=42):
    """
    target_task: 'syntax' for Task A, 'pragma' for Task B.
    """
    set_seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # اختيار العمود المستهدف بناءً على المهمة المطلوبة
    label_col = 'label_syntax' if target_task == 'syntax' else 'label_pragma'
    num_labels = 2 # لأننا قمنا بدمج الفئات ثنائياً (Binary)

    def tokenize_function(examples):
        return tokenizer(examples["clean_text"], truncation=True, padding="max_length", max_length=128)

    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=seed)
    
    fold_results = []
    oof_predictions = np.zeros(len(df))
    oof_true = np.zeros(len(df))
    oof_probs = np.zeros((len(df), num_labels))
    
    total_train_time = 0
    total_inf_time = 0

    for fold, (train_val_idx, test_idx) in enumerate(skf.split(df['clean_text'], df[label_col])):
        print(f"\n--- Training Fold {fold+1}/{k_folds} [TASK: {target_task.upper()}] ---")
        
        train_val_df = df.iloc[train_val_idx].reset_index(drop=True)
        test_df = df.iloc[test_idx].reset_index(drop=True)
        
        # 15% Validation من بيانات التدريب لاختيار أفضل Checkpoint
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            train_val_df['clean_text'], train_val_df[label_col], 
            test_size=0.15, stratify=train_val_df[label_col], random_state=seed
        )
        
        train_df = pd.DataFrame({'clean_text': train_texts, 'label': train_labels}).reset_index(drop=True)
        val_df = pd.DataFrame({'clean_text': val_texts, 'label': val_labels}).reset_index(drop=True)
        
        train_ds = Dataset.from_pandas(train_df[['clean_text', 'label']]).map(tokenize_function, batched=True)
        val_ds = Dataset.from_pandas(val_df[['clean_text', 'label']]).map(tokenize_function, batched=True)
        test_ds = Dataset.from_pandas(test_df[['clean_text', label_col]].rename(columns={label_col: 'label'})).map(tokenize_function, batched=True)

        # حساب أوزان الفئات للتعامل مع عدم التوازن (Imbalance)
        classes = np.unique(train_df['label'])
        weights = compute_class_weight(class_weight='balanced', classes=classes, y=train_df['label'])

        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=num_labels, problem_type="single_label_classification"
        )

        training_args = TrainingArguments(
            output_dir=f"./results_{target_task}_fold_{fold}",
            eval_strategy="epoch",
            save_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=8,
            num_train_epochs=5,
            load_best_model_at_end=True,
            metric_for_best_model="macro_f1", # الاعتماد على Macro-F1 لاختيار النموذج الأفضل
            save_total_limit=1,
            report_to="none"
        )

        trainer = WeightedTrainer(
            class_weights=weights,
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            compute_metrics=compute_metrics,
        )

        t0 = time.time()
        trainer.train()
        total_train_time += (time.time() - t0)

        t1 = time.time()
        test_preds = trainer.predict(test_ds)
        eval_res = test_preds.metrics
        total_inf_time += (time.time() - t1)
        
        cleaned_eval_res = {
            'eval_macro_f1': eval_res['test_macro_f1'],
            'eval_accuracy': eval_res['test_accuracy'],
            'eval_precision': eval_res['test_precision_macro'],
            'eval_recall': eval_res['test_recall_macro']
        }
        fold_results.append(cleaned_eval_res)
        
        # OOF Saving
        oof_predictions[test_idx] = np.argmax(test_preds.predictions, axis=-1)
        oof_true[test_idx] = test_df[label_col].values
        oof_probs[test_idx] = torch.nn.functional.softmax(torch.tensor(test_preds.predictions), dim=-1).numpy()

    # Aggregate Metrics
    f1_scores = [r['eval_macro_f1'] for r in fold_results]
    acc_scores = [r['eval_accuracy'] for r in fold_results]

    metrics = {
        'Macro_F1': f"{np.mean(f1_scores)*100:.2f} ±{np.std(f1_scores)*100:.2f}",
        'Accuracy': f"{np.mean(acc_scores)*100:.2f} ±{np.std(acc_scores)*100:.2f}",
        'Train_Time(s)': f"{total_train_time:.1f}",
        'Inf_Time(s)': f"{total_inf_time:.2f}"
    }

    return metrics, fold_results, oof_predictions, oof_true, oof_probs
