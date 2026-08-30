from django.contrib import admin
from .models import SentimentFeedback

# Customize list display for SentimentFeedback
class SentimentFeedbackAdmin(admin.ModelAdmin):
    list_display = ("text", "preprocessed_text", "true_label", "predicted_label", "timestamp")
    search_fields = ("text", "preprocessed_text")  # optional, adds search box
    list_filter = ("true_label", "predicted_label", "timestamp")  # optional, adds filters

# Register models
admin.site.register(SentimentFeedback, SentimentFeedbackAdmin)

# class SentimentFeedbackArchiveAdmin(admin.ModelAdmin):
#     list_display = ("text", "preprocessed_text", "true_label", "predicted_label", "timestamp")

# admin.site.register(SentimentFeedbackArchive, SentimentFeedbackArchiveAdmin)
