
import nltk
from flask import Flask, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  #16MB


nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')


pos_full_forms = {
    'CC': 'Coordinating conjunction',
    'CD': 'Cardinal number',
    'DT': 'Determiner',

'EX': 'Existential there',
'FW': 'Foreign word',
'IN': 'Preposition or subordinating conjunction',
'JJ': 'Adjective',
'JJR': 'Adjective, comparative',
'JJS': 'Adjective, superlative',
'LS': 'List item marker',
'MD': 'Modal',
'NN': 'Noun, singular or mass',
'NNS': 'Noun, plural',
'NNP': 'Proper noun, singular',
'NNPS': 'Proper noun, plural',
'PDT': 'Predeterminer',
'POS': 'Possessive ending',
'PRP': 'Personal pronoun',
'PRP$': 'Possessive pronoun',
'RB': 'Adverb',
'RBR': 'Adverb, comparative',
'RBS': 'Adverb, superlative',
'RP': 'Particle',
'SYM': 'Symbol',
'TO': 'to',
'UH': 'Interjection',
'VB': 'Verb, base form',
'VBD': 'Verb, past tense',
'VBG': 'Verb, gerund or present participle',
'VBN': 'Verb, past participle',
'VBP': 'Verb, non-3rd person singular present',
'VBZ': 'Verb, 3rd person singular present',
'WDT': 'Wh-determiner',
'WP': 'Wh-pronoun',
'WP$': 'Possessive wh-pronoun',
'WRB': 'Wh-adverb'
   
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_keywords', methods=['POST'])
def get_keywords():
    uploaded_file = request.files['file']
    if uploaded_file.filename == '':
        return "No file selected"

    file_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(uploaded_file.filename))
    uploaded_file.save(file_path)

    try:
        with open(file_path, 'r') as file:
            text = file.read()
    except Exception as e:
        return f"An error occurred: {e}"

    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
    filtered_text = ' '.join(filtered_tokens)

    vectorizer = TfidfVectorizer()
    tfidf_scores = vectorizer.fit_transform([filtered_text])
    feature_names = vectorizer.get_feature_names_out()
    feature_names = feature_names.tolist()
    sorted_terms = [term for term in feature_names]
    sorted_terms.sort(key=lambda x: tfidf_scores[0, feature_names.index(x)], reverse=True)

    N = 10
    top_keywords = sorted_terms[:N]

    
    tokens_with_pos = nltk.pos_tag(filtered_tokens)
    top_keywords_with_pos = []
    for keyword in top_keywords:
        for token, pos_tag in tokens_with_pos:
            if keyword == token:
                full_pos_tag = pos_full_forms.get(pos_tag, pos_tag)  
                top_keywords_with_pos.append((keyword, full_pos_tag))
                break

    return render_template('index.html', keywords=top_keywords_with_pos)

if __name__ == '__main__':
    app.run(debug=True)

