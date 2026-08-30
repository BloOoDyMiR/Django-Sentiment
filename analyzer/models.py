from django.db import models

class SentimentFeedback(models.Model):
    LABEL_CHOICES = (
        (0, "خوشحال"),
        (1, "ناراحت"),
    )

    text = models.TextField(verbose_name="متن اصلی")
    preprocessed_text = models.TextField(verbose_name="متن پردازش‌شده")
    true_label = models.IntegerField(verbose_name="برچسب واقعی", choices=LABEL_CHOICES)
    predicted_label = models.IntegerField(verbose_name="برچسب پیش‌بینی‌شده", choices=LABEL_CHOICES)
    timestamp = models.DateTimeField(auto_now_add=True, verbose_name="تاریخ ثبت")

    class Meta:
        verbose_name = "بازخورد تحلیل احساس"
        verbose_name_plural = "بازخوردهای تحلیل احساس"

    def __str__(self):
        true = dict(self.LABEL_CHOICES).get(self.true_label, "?")
        pred = dict(self.LABEL_CHOICES).get(self.predicted_label, "?")
        return f"{self.text[:50]}... | واقعی: {true} پیش‌بینی: {pred}"



# class SentimentFeedbackArchive(models.Model):
#     text = models.TextField(verbose_name="متن اصلی")
#     preprocessed_text = models.TextField(verbose_name="متن پیش‌پردازش شده")
#     true_label = models.IntegerField(verbose_name="برچسب واقعی")
#     predicted_label = models.IntegerField(verbose_name="برچسب پیش‌بینی")
#     timestamp = models.DateTimeField(verbose_name="زمان ثبت")
#     archived_at = models.DateTimeField(auto_now_add=True, verbose_name="زمان بایگانی")

#     def __str__(self):
#         return f"{self.text[:50]}... | True: {self.true_label} Pred: {self.predicted_label}"

#     class Meta:
#         verbose_name = "بایگانی تحلیل احساس"
#         verbose_name_plural = "بایگانی بازخوردهای تحلیل احساس"
