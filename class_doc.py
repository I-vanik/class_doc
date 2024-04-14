import textract
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
import shutil

# Функция для извлечения текста из документа
def extract_text(file_path):
    text = textract.process(file_path).decode('utf-8')
    return text

# Функция для определения класса документа с помощью k-nn
def classify_document(text):
    documents = ['доверенность', 'договор', 'акт', 'заявление', 'приказ', 'счет', 'приложение', 'соглашение', 'устав', 'договор оферты', 'решение']
    
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform([text])
    
    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(X, [0])
    
    prediction = knn.predict(X)
    
    return documents[int(prediction[0])]

# Папка с исходными документами
source_folder = 'путь_к_папке_с_документами'

# Создание папок для классов документов
for document_class in ['доверенность', 'договор', 'акт', 'заявление', 'приказ', 'счет', 'приложение', 'соглашение', 'устав', 'договор оферты', 'решение']:
    os.makedirs(document_class, exist_ok=True)

# Обработка каждого документа и классификация
for file_name in os.listdir(source_folder):
    file_path = os.path.join(source_folder, file_name)
    text = extract_text(file_path)
    document_class = classify_document(text)
    
    # Перемещение документа в соответствующую папку
    shutil.move(file_path, document_class)

# Создание папки для результатов на рабочем столе
os.makedirs(os.path.expanduser('~/Desktop/классифицированные_документы'), exist_ok=True)

# Перемещение папок с классами документов на рабочий стол
for document_class in ['доверенность', 'договор', 'акт', 'заявление', 'приказ', 'счет', 'приложение', 'соглашение', 'устав', 'договор оферты', 'решение']:
    shutil.move(document_class, os.path.expanduser('~/Desktop/классифицированные_документы'))