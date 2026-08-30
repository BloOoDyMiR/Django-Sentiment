from django import forms

class SentimentForm(forms.Form):
    text = forms.CharField(
        label="متن شما",
        widget=forms.Textarea(attrs={"rows": 4, "placeholder": "اینجا متن را وارد کنید"})
    )

LABEL_CHOICES = [(0, "HAPPY"), (1, "SAD")]

class FeedbackForm(forms.Form):
    text = forms.CharField(widget=forms.Textarea(attrs={"rows":4}))
    correct_label = forms.ChoiceField(choices=LABEL_CHOICES, widget=forms.RadioSelect)
