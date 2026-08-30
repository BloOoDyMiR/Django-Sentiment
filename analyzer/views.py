from django.shortcuts import render, redirect
from .forms import SentimentForm, FeedbackForm
from .ml_service import predict_sentiment, vectorizer, model, scaler, preprocess_text, extract_multi_aspects
from .models import SentimentFeedback
import scipy.sparse as sp
import numpy as np

# views.py (home view)
def home(request):
    result = None
    analyzed_text = None
    if request.method == "POST":
        try:
            text = request.POST.get("text")
            if not text:
                return render(request, "analyzer/home.html", {
                    "error": "لطفاً متن وارد کنید",
                    "feedback_form": FeedbackForm()
                })
            processed_text = preprocess_text(text)
            vec = vectorizer.transform([processed_text])
            aspect_feat = np.array([extract_multi_aspects(processed_text)])
            aspect_feat_scaled = scaler.transform(aspect_feat)
            X_new = sp.hstack([vec, sp.csr_matrix(aspect_feat_scaled)])
            label_id = model.predict(X_new)[0]
            label_map = {0: "HAPPY", 1: "SAD"}
            result = label_map[label_id]
            analyzed_text = text  # Keep the original text to send to feedback form
        except Exception as e:
            return render(request, "analyzer/home.html", {
                "error": f"خطا در تحلیل: {str(e)}",
                "feedback_form": FeedbackForm()
            })

    feedback_form = FeedbackForm(initial={"text": analyzed_text})
    return render(request, "analyzer/home.html", {
        "result": result,
        "feedback_form": feedback_form,
        "analyzed_text": analyzed_text
    })

def feedback(request):
    if request.method == "POST":
        try:
            text = request.POST.get("text")  # this is the actual analyzed text
            correct_label = request.POST.get("correct_label")  # 'yes' or 'no'

            processed_text = preprocess_text(text)
            vec = vectorizer.transform([processed_text])
            aspect_feat = np.array([extract_multi_aspects(processed_text)])
            aspect_feat_scaled = scaler.transform(aspect_feat)
            X_new = sp.hstack([vec, sp.csr_matrix(aspect_feat_scaled)])
            predicted_label = int(model.predict(X_new)[0])

            # Determine true label
            if correct_label == "yes":
                true_label = predicted_label
            else:
                # Flip the label
                true_label = 1 if predicted_label == 0 else 0

            # Save to DB
            SentimentFeedback.objects.create(
                text=text,
                preprocessed_text=processed_text,
                predicted_label=predicted_label,
                true_label=true_label
            )
            return redirect("home")
        except Exception as e:
            return render(request, "analyzer/home.html", {
                "error": f"خطا در ذخیره بازخورد: {str(e)}",
                "feedback_form": FeedbackForm()
            })
    return redirect("home")