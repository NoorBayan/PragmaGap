import torch
from transformers import Trainer
import torch.nn as nn

class WeightedTrainer(Trainer):
    def __init__(self, class_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # تحويل الأوزان إلى Tensor وإرسالها لاحقاً إلى الـ Device المناسب
        self.class_weights = torch.tensor(class_weights, dtype=torch.float32) if class_weights is not None else None

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        
        # استخدام دالة الخسارة العادية مع تمرير أوزان الفئات
        loss_fct = nn.CrossEntropyLoss(weight=self.class_weights.to(self.args.device) if self.class_weights is not None else None)
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        
        return (loss, outputs) if return_outputs else loss
