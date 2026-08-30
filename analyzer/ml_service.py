import os
import re
import joblib
import numpy as np
import scipy.sparse as sp
from django.conf import settings
from hazm import Normalizer, sent_tokenize

ARTIFACT_DIR = os.path.join(os.path.dirname(__file__), "ml")
MODEL_PATH = os.path.join(ARTIFACT_DIR, "sentiment_model.pkl")
VECT_PATH = os.path.join(ARTIFACT_DIR, "tfidf_vectorizer.pkl")
SCALER_PATH = os.path.join(ARTIFACT_DIR, "scaler.pkl")

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECT_PATH)
scaler = joblib.load(SCALER_PATH)

normalizer = Normalizer()

PERSIAN_STOPWORDS = {
    'و', 'در', 'به', 'از', 'که', 'را', 'این', 'آن', 'می', 'ها',
    'برای', 'با', 'تا', 'اما', 'اگر', 'یک', 'یا', 'نیز', 'هم',
    'بر', 'شود', 'بود', 'است', 'کرد', 'کردن', 'شد', 'داشت',
    'دارد', 'دیگر', 'هر', 'همه', 'بین', 'زیرا', 'چون', 'باید',
    'هیچ', 'مثل', 'چند', 'چرا', 'کجا', 'چطور', 'بنابراین',
    'قبل', 'بعد', 'هنوز', 'اگرچه', 'بدون', 'زیاد', 'کم', 'حتی',
    'وقتی', 'وقتی‌که', 'مانند', 'کن', 'کنند', 'کنید', 'کند',
    'ما', 'من', 'تو', 'او', 'آنها', 'ماها', 'شما', 'خود', 'خودش',
    'خودم', 'خودت', 'خودمان', 'خودشان', 'خودشون'
}

positive_words = set([
    "عالی", "فوقالعاده", "زیبا", "خوب", "مثبت", "شاد", "خوشحال", "خوشایند",
    "دوستداشتنی", "عالی", "بینظیر", "ستودنی", "محشر", "کمنظیر", "تحسینبرانگیز",
    "قشنگ", "چشمنواز", "دلانگیز", "دلپذیر", "خوشمزه", "لذیذ", "مطبوع",
    "پرانرژی", "سرزنده", "شاداب", "سرحال", "باانرژی", "نشاطانگیز",
    "قوی", "مقتدر", "توانا", "کارآمد", "موثر", "مفید", "سودمند",
    "هوشمند", "زیرک", "باهوش", "نابغه", "خلاق", "مبتکر", "نوآور",
    "مهربان", "دلرحم", "باکرامت", "باحیا", "بااخلاق", "خوشقلب", "نیکوکار",
    "صمیمی", "صادق", "رک", "امین", "قابلاعتماد", "پاکدامن", "درستکار",
    "سخاوتمند", "بخشنده", "دستودلباز", "کریم", "بلندهمت",
    "شجاع", "نترس", "بیباک", "دلیر", "قات", "جرئتمند",
    "صبور", "شکیبا", "بردبار", "آرام", "مطمئن", "امیدوار", "خوشبین",
    "موفق", "کامیاب", "پیروز", "فاتح", "سربلند", "رستگار",
    "ثروتمند", "دارا", "متمول", "باسعادت", "خوشبخت", "خوشقدم",
    "مشهور", "نامدار", "معروف", "پرآوازه", "برجسته", "درخشنده",
    "آرامشبخش", "مسحورکننده", "فریبنده", "جذاب", "گیرا", "تماشایی",
    "تمیز", "پاکیزه", "خالص", "ناب", "اصیل", "واقعی", "حقیقی",
    "نرم", "لطیف", "ملایم", "خوشآوا", "خوشاینگ", "مهماننواز",
    "فداکار", "ایثارگر", "ازخودگذشته", "دلسوز", "غیرتمند", "میهندوست",
    "ساده", "بیپیرایه", "بیآلایش", "متواضع", "فروتن", "بیادعا",
    "بااستعداد", "باذوق", "باشعور", "خردمند", "دانا", "عاقل", "فرزانہ",
    "تازه", "نو", "جدید", "بکر", "منحصربهفرد", "ویژه", "مخصوص",
    "ارزنده", "گرانبها", "گرانقدر", "عزیز", "محترم", "حرمتدار",
    "مطلوب", "موردعلاقه", "محبوب", "مردمی", "محبوب",
    "مطمئن", "مسلط", "مسلط", "چیره", "حرفهای", "استاد", "ماہر",
    "خوشقدم", "خوششانس", "مبارک", "میخکوبکننده", "حیرتانگیز", "شگفتانگیز"
])

negative_words = set([
    "بد", "زشت", "زشت", "ناچیز", "بیارزش", "پست", "منفور", "مضحکه",
    "غمگین", "اندوهگین", "افسرده", "دلگیر", "محزون", "ناراحت", "غمانگیز",
    "عصبانی", "خشمگین", "خشماگین", "برآشفته", "خروشان", "قیافه‌گیر",
    "متنفر", "مستاصل", "ناامید", "دلسرد", "دلزده", "شکست‌خورده",
    "ترسیده", "وحشت‌زده", "هراسان", "مضطرب", "نگران", "پریشان",
    "خسته", "فرسوده", "بیحال", "کسل", "ملول", "رنجور",

    "احمق", "نادان", "بیخرد", "کودن", "خر", "گاو", "خل", "پست‌فطرت",
    "بیعرضه", "نالایق", "بیمصرف", "تنبل", "خراب", "کثیف", "چرک",
    "زشت", "زشت‌رو", "کریه", "بیشرف", "بیناموس", "حرامزاده", "لاشه",
    "دزد", "کلاهبردار", "حقه‌باز", "فریبکار", "خیانتکار", "مزدور",

    "فتحه", "مشکل", "مصیبت", "فاجعه", "بدبختی", "بیچارگی", "رنج", "عذاب",
    "درد", "رنجش", "نفرت", "کینه", "حسادت", "شک", "ترس", "وحشت",
    "جنگ", "نزاع", "درگیری", "حماقت", "اشتباه", "گناه", "تقصیر",
    "خرابی", "ویرانی", "نابودی", "مرگ", "قتل", "جنایت", "خون", "اشک",

    "بیفایده", "باطل", "پوچ", "مزخرف", "مسخره", "خندهدار", "هولناک",
    "وحشتناک", "ترسناک", "نفرتانگیز", "تهوع‌آور", "کسل‌کننده", "خفقان‌آور",
    "طاقت‌فرسا", "خسته‌کننده", "غمبار", "سیاه", "تاریک", "تیره", "تلخ",
    "سخت", "دشوار", "طاقت‌سوز", "گران", "سنگین", "کند", "کهنه", "پوسیده",

    "میکشد", "می‌میرد", "گریه می‌کند", "نفرت دارد", "عصبانی است", "نابود می‌کند",
    "دزدی می‌کند", "فریب می‌دهد", "خیانت می‌کند", "شکست می‌خورد", "وامی‌ماند",
    "گیر می‌کند", "عذاب می‌دهد", "اذیت می‌کند", "تحقیر می‌کند", "مسخره می‌کند"
])

delivery_words = set(["دیر", "سریع", "زمان", "ارسال", "پیک"])


def preprocess_text(text: str) -> str:
    """
    Preprocess Persian text using hazm library.
    This replaces the old simple preprocessing with a more robust approach.
    """
    text = normalizer.normalize(text)
    
    text = text.replace('\u200c', '')
    
    text = re.sub(r'[A-Za-z0-9]+', ' ', text)
    
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    sentences = sent_tokenize(text)
    
    cleaned_sentences = []
    for s in sentences:
        tokens = s.split()
        tokens = [t for t in tokens if t not in PERSIAN_STOPWORDS]
        cleaned_sentences.append(" ".join(tokens))
    
    return " ".join(s for s in cleaned_sentences if s.strip())


def extract_multi_aspects(text: str) -> list:
    """
    Extract aspect features from preprocessed text.
    Counts positive, negative, and delivery-related words.
    """
    tokens = text.split()
    pos_count = sum(t in positive_words for t in tokens)
    neg_count = sum(t in negative_words for t in tokens)
    delivery_count = sum(t in delivery_words for t in tokens)
    return [pos_count, neg_count, delivery_count]


LABEL_MAP = {0: "HAPPY", 1: "SAD"}


def predict_sentiment(text: str) -> str:
    """
    Predict sentiment for given Persian text.
    
    Args:
        text: Raw Persian text input
        
    Returns:
        Sentiment label: "HAPPY" or "SAD"
    """
    processed = preprocess_text(text)
    
    vec = vectorizer.transform([processed])
    
    aspect_feat = np.array([extract_multi_aspects(processed)])
    aspect_feat_scaled = scaler.transform(aspect_feat)
    
    X_new = sp.hstack([vec, sp.csr_matrix(aspect_feat_scaled)])
    
    label_id = int(model.predict(X_new)[0])
    return LABEL_MAP[label_id]