import re
import string
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer

snowball = SnowballStemmer('english')
lemmatizer = WordNetLemmatizer()

# Clean punctuation and multiple spaces
def clean_sentence(text):
    clean_punctuation = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
    trim_spaces_text = re.sub(r"\s(?=\s)", "", clean_punctuation)
    return trim_spaces_text.lower()

# lowercase data and clean punctuation and spaces
def clean_data(data):
    return [clean_sentence(sentence) for sentence in data]

# tokenize data and clean stopwords
def filter_data(data):
    tokenized_data = [word_tokenize(sentence) for sentence in data]
    stop_words = set(stopwords.words('english'))
    return [[word for word in sentence if word not in stop_words] for sentence in tokenized_data]

# lemmatize received data
def lemmatize_data(data):
    return [[lemmatizer.lemmatize(word) for word in sentence] for sentence in data]

# preprocess received text data
def preprocess_data(data):
    return lemmatize_data(filter_data(clean_data(data)))

# join data tokens
def join_sentences(data):
    return [" ".join(sentence_words) for sentence_words in data]

# Get data preprocessing attributes ['content', 'tokens', 'preprocessed_text']
def get_preprocessing_attributes(data):
    data['content'] = data.title + " " + data.text
    data['tokens'] = preprocess_data(data['content'])
    data['preprocessed_text'] = join_sentences(data['tokens'])
    return data