# Django Sentiment Analysis

A Django-based web application that performs sentiment analysis using Natural Language Processing (NLP).

This project demonstrates how machine learning and web development can be combined to create an interactive sentiment analysis system where users can submit text and receive sentiment predictions.

## Features

- Text sentiment classification
- Django web interface
- NLP-based text processing
- Machine learning model integration
- User-friendly input form
- Real-time sentiment prediction
- Easy to extend with new models and datasets

## Project Overview

The application receives text input from users and processes it through a sentiment analysis pipeline.

The general workflow:

```
User Input
    |
    ↓
Django Form
    |
    ↓
Text Preprocessing
    |
    ↓
Sentiment Analysis Model
    |
    ↓
Prediction Result
```

The system can classify text into sentiment categories such as:

- Positive
- Negative
- Neutral

## Technologies Used

### Backend

- Python
- Django

### Machine Learning / NLP

- Natural Language Processing
- Machine Learning models
- Text preprocessing techniques

### Frontend

- HTML
- CSS
- Django Templates

### Database

- SQLite (default Django database)

## Installation

Clone the repository:

```bash
git clone https://github.com/BloOoDyMiR/Django-Sentiment.git
```

Move into the project directory:

```bash
cd Django-Sentiment
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate the environment:

### Windows

```bash
venv\Scripts\activate
```

### Linux / macOS

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Database Setup

Run Django migrations:

```bash
python manage.py migrate
```

## Running the Application

Start the development server:

```bash
python manage.py runserver
```

Open your browser:

```
http://127.0.0.1:8000/
```

## Usage

1. Open the web application.
2. Enter a text message.
3. Submit the form.
4. The system analyzes the text.
5. The predicted sentiment is displayed.

## Project Structure

```
Django-Sentiment/
│
├── manage.py
├── requirements.txt
│
├── app/
│   ├── views.py
│   ├── models.py
│   ├── forms.py
│
├── templates/
│
├── static/
│
└── README.md
```

## Machine Learning Pipeline

The sentiment analysis process includes:

- Text cleaning
- Tokenization
- Feature extraction
- Model prediction
- Result visualization

The model can be replaced or improved with advanced NLP approaches such as:

- TF-IDF + Machine Learning classifiers
- LSTM networks
- GRU networks
- Transformer-based models (BERT, RoBERTa)

## Future Improvements

Possible improvements:

- Add user authentication
- Add sentiment history dashboard
- Support multiple languages
- Deploy using Docker
- Add REST API endpoints
- Improve accuracy with transformer models
- Add data visualization charts

## Contributing

Contributions are welcome.

Feel free to open issues or submit pull requests to improve the project.

## License

This project is open source and available under the MIT License.

## Author

Created by [BloOoDyMiR](https://github.com/BloOoDyMiR)
