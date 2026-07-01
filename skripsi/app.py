import os
from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify, send_from_directory
from markupsafe import Markup
from werkzeug.utils import secure_filename
import pymysql
import binascii
import csv
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd
import re
import string
import seaborn as sns
import matplotlib
matplotlib.use("Agg")  # Gunakan backend non-GUI sebelum mengimpor pyplot
import matplotlib.pyplot as plt
from io import BytesIO
import base64

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
import logging
import time
import tweepy
import requests
from googletrans import Translator
from bs4 import BeautifulSoup
import unidecode
from deep_translator import GoogleTranslator
from sklearn.multiclass import OneVsRestClassifier
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import joblib
import json
import torch
import numpy as np
from sklearn.preprocessing import StandardScaler
from collections import Counter


nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')


app = Flask(__name__, template_folder='templates')
app.secret_key = binascii.hexlify(os.urandom(16))
# Konfigurasi MySQL untuk XAMPP
app.config['MYSQL_DATABASE_USER'] = 'root'
app.config['MYSQL_DATABASE_DB'] = 'piter-skripsi'
app.config['MYSQL_DATABASE_PASSWORD'] = ''
app.config['MYSQL_DATABASE_HOST'] = 'localhost'
app.config['UPLOAD_FOLDER'] = 'static'

translator = GoogleTranslator(
    source='auto',
    target='en'
)

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# ============================================================
# LOAD BERT MODEL
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("\n[INFO] Loading BERT...")

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
bert = AutoModel.from_pretrained("bert-base-uncased")

bert.to(device)
bert.eval()

print("[DONE] BERT ready")



# ============================================================
# CONFIG
# ============================================================
MAX_LEN = 128
BATCH_SIZE = 8   # lebih aman untuk RAM
# ============================================================


# ============================================================
# FUNCTION EMBEDDING
# ============================================================
def get_batch_embedding(text_list):
    inputs = tokenizer(
        text_list,
        padding=True,
        truncation=True,
        max_length=MAX_LEN,
        return_tensors="pt"
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = bert(**inputs)

    # CLS token embedding
    cls_embeddings = outputs.last_hidden_state[:, 0, :]

    return cls_embeddings.cpu().numpy().astype(np.float32)

# load scaler (GLOBAL sebaiknya di atas app.py)
scaler = joblib.load("models/embedding_scaler.pkl")

ALLOWED_EXTENSIONS = {'csv','xlsx','xls'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def get_db_connection():
    try:
        connection = pymysql.connect(
            host=app.config['MYSQL_DATABASE_HOST'],
            user=app.config['MYSQL_DATABASE_USER'],
            password=app.config['MYSQL_DATABASE_PASSWORD'],
            db=app.config['MYSQL_DATABASE_DB'],
            cursorclass=pymysql.cursors.DictCursor
        )
        return connection
    except Exception as e:
        print("Error: {e}")
        flash(f"Error: {e}", 'danger')
        return None

# ==========================
# LOAD MODEL SEKALI (PENTING)
# ==========================
ada_model = joblib.load("models/adaboost_nb.pkl")
scaler = joblib.load("models/embedding_scaler.pkl")

models = ada_model["models"]
alphas = ada_model["alphas"]
classes = models[0].classes_

def preprocess_text(text):

    # ==========================
    # LOWERCASE
    # ==========================
    text = text.lower()

    # ==========================
    # CLEANSING
    # ==========================
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()

    # ==========================
    # TOKENIZATION
    # ==========================
    tokens = word_tokenize(text)

    # ==========================
    # STOPWORD REMOVAL
    # ==========================
    tokens = [
        word
        for word in tokens
        if word not in stop_words
    ]

    # ==========================
    # LEMMATIZATION
    # ==========================
    tokens = [
        lemmatizer.lemmatize(word)
        for word in tokens
    ]

    # Gabung kembali menjadi kalimat
    clean_text = " ".join(tokens)

    return clean_text

@app.route('/', methods=['GET', 'POST'])
def review_sentiment():

    prediction = None
    input_text = ""

    try:
        if request.method == 'POST':

            # ==========================
            # INPUT
            # ==========================
            input_text = request.form.get('text', '')

            if not input_text.strip():
                flash("Text tidak boleh kosong", "warning")
                return render_template("index.html", prediction=None, input_text="")

            # ==========================
            # TRANSLATE
            # ==========================
            try:
                translated_text = translator.translate(input_text)
            except Exception:
                translated_text = input_text

            # ==========================
            # PREPROCESS
            # ==========================
            clean_text = preprocess_text(translated_text)

            # ==========================
            # EMBEDDING
            # ==========================
            X = np.array(get_batch_embedding([clean_text]))

            # ==========================
            # LOAD MODEL + ENCODER + SCALER
            # ==========================
            scaler = joblib.load("models/embedding_scaler.pkl")

            bundle = joblib.load("models/adaboost_nb.pkl")
            models = bundle["models"]
            alphas = bundle["alphas"]
            encoder = bundle["encoder"]

            classes = encoder.classes_

            # ==========================
            # SCALE
            # ==========================
            X = scaler.transform(X)

            # ==========================
            # ADABOOST PREDICTION (FIXED)
            # ==========================
            final_score = np.zeros((1, len(classes)))

            for alpha, model in zip(alphas, models):

                # 🔥 FIX: pakai probability (bukan predict)
                proba = model.predict_proba(X)[0]

                final_score[0] += alpha * proba

            # ==========================
            # FINAL RESULT
            # ==========================
            y_pred = np.argmax(final_score, axis=1)[0]
            prediction = encoder.inverse_transform([y_pred])[0]

    except Exception as e:
        flash(f"Error: {str(e)}", "danger")

    return render_template(
        "index.html",
        prediction=prediction,
        input_text=input_text
    )

@app.route('/data_visualization', methods=['GET'])
def data_visualization():

    connection = get_db_connection()

    try:
        with connection.cursor() as cursor:

            # =====================================
            # AMBIL DATA REVIEW
            # =====================================
            cursor.execute("""
                SELECT
                    d.id_dataset,
                    d.review_id,
                    d.clean_text,
                    d.sentiment
                FROM dataset d
                ORDER BY d.id_dataset
                LIMIT 300
            """)

            rows = cursor.fetchall()

        # =====================================
        # KONVERSI DATA
        # =====================================
        ids = []
        texts = []
        true_labels = []

        for r in rows:
            ids.append(r["review_id"])
            texts.append(r["clean_text"])
            true_labels.append(r["sentiment"])

        df = pd.DataFrame({
            "id": ids,
            "text": texts,
            "true": true_labels
        })

        # =====================================
        # LOAD MODEL
        # =====================================
        scaler = joblib.load("models/embedding_scaler.pkl")
        bundle = joblib.load("models/adaboost_nb.pkl")

        models = bundle["models"]
        alphas = bundle["alphas"]
        encoder = bundle["encoder"]

        classes = encoder.classes_

        # =====================================
        # EMBEDDING
        # =====================================
        embeddings = get_batch_embedding(df["text"].tolist())
        X = scaler.transform(np.array(embeddings))

        # =====================================
        # PREDIKSI NAIVE BAYES + ADABOOST
        # =====================================
        final_score = np.zeros((X.shape[0], len(classes)))

        for alpha, model in zip(alphas, models):
            proba = model.predict_proba(X)
            final_score += alpha * proba

        y_pred = np.argmax(final_score, axis=1)
        pred_labels = encoder.inverse_transform(y_pred)

        # =====================================
        # HASIL AKHIR
        # =====================================
        df["ground_truth"] = df["true"]
        df["nb_ada"] = pred_labels

        # =====================================
        # HITUNG JUMLAH DATA
        # =====================================
        gt_counts = df["ground_truth"].value_counts().to_dict()
        nb_counts = df["nb_ada"].value_counts().to_dict()

        labels = ["Negative", "Neutral", "Positive"]

        gt_values = [gt_counts.get(label, 0) for label in labels]
        nb_values = [nb_counts.get(label, 0) for label in labels]

        return render_template(
            "data_visualization.html",
            tables=df.to_dict(orient="records"),
            labels=labels,
            gt_values=gt_values,
            nb_values=nb_values
        )

    finally:
        connection.close()
    
@app.route('/login', methods=['GET', 'POST'])
def login():
    if 'user_id' in session:
        return redirect(url_for('dashboard'))
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        connection = get_db_connection()
        if connection is None:
            return "Error: Unable to connect to MySQL"
        try:
            with connection.cursor() as cur:
                cur.execute("SELECT * FROM users WHERE email=%s AND password=MD5(%s)", (email, password))
                user = cur.fetchone()
                if user:
                    session['user_id'] = user['id']
                    session['username'] = user['nama_lengkap']
                    return redirect(url_for('dataset'))
                    print("Berhasil")
                else:
                    flash(f'Invalid email or password', 'danger')
                    return redirect(url_for('login'))
        except Exception as e:
            print("Error: {e}")
            flash(f"Error: {e}", 'danger')
        finally:
            connection.close()
    return render_template('login.html')

@app.route('/dataset/embedding/<int:id_dataset>')
def detail_embedding(id_dataset):

    if 'user_id' not in session:
        return jsonify({"error": "Unauthorized"}), 401

    connection = get_db_connection()

    try:
        with connection.cursor() as cursor:
            cursor.execute("""
                SELECT embedding
                FROM dataset_embedding
                WHERE id_dataset = %s
            """, (id_dataset,))

            data = cursor.fetchone()

            if not data:
                return jsonify({"embedding": ""})

            return jsonify({
                "embedding": data["embedding"]
            })

    finally:
        connection.close()


@app.route('/api/dataset')
def api_dataset():

    draw = request.args.get('draw', type=int, default=1)
    start = request.args.get('start', type=int, default=0)
    length = request.args.get('length', type=int, default=10)

    connection = get_db_connection()

    try:
        with connection.cursor() as cursor:

            # Total data
            cursor.execute("SELECT COUNT(*) AS total FROM dataset")
            total = cursor.fetchone()["total"]

            # Data per halaman
            cursor.execute("""
                SELECT
                    d.id_dataset,
                    d.review_id,
                    d.content,
                    d.sentiment,
                    CASE
                        WHEN de.id_dataset IS NOT NULL THEN 1
                        ELSE 0
                    END AS has_embedding
                FROM dataset d
                LEFT JOIN dataset_embedding de
                    ON d.id_dataset = de.id_dataset
                ORDER BY d.id_dataset
                LIMIT %s OFFSET %s
            """, (length, start))

            rows = cursor.fetchall()

        return jsonify({
            "draw": draw,
            "recordsTotal": total,
            "recordsFiltered": total,
            "data": rows
        })

    finally:
        connection.close()

@app.route('/dataset')
def dataset():

    if 'user_id' not in session:
        flash("Please log in first.", "warning")
        return redirect(url_for('login'))

    return render_template(
        'dataset.html',
        username=session['username']
    )

@app.route('/tambah_dataset', methods=['GET', 'POST'])
def tambah_dataset():

    if 'user_id' not in session:
        flash("Please log in first.", "warning")
        return redirect(url_for('login'))

    if request.method == 'POST':

        review_id = request.form['review_id']
        content = request.form['content']
        score = request.form['score']
        sentiment = request.form['sentiment']

        translated_text = translator.translate(content)

        clean_text = preprocess_text(translated_text)

        connection = get_db_connection()

        try:

            with connection.cursor() as cursor:
                
                # CEK DUPLIKAT (taruh di sini)
                cursor.execute("""
                    SELECT id_dataset
                    FROM dataset
                    WHERE review_id = %s
                """, (review_id,))

                existing = cursor.fetchone()

                if existing:
                    flash("Review ID sudah digunakan!", "danger")
                    return redirect(url_for('tambah_dataset'))
                # =========================
                # INSERT DATASET
                # =========================
                cursor.execute("""
                    INSERT INTO dataset
                    (
                        review_id,
                        content,
                        score,
                        translated_text,
                        clean_text,
                        sentiment
                    )
                    VALUES
                    (%s, %s, %s, %s, %s, %s)
                """, (
                    review_id,
                    content,
                    score,
                    translated_text,
                    clean_text,
                    sentiment
                ))

                id_dataset = cursor.lastrowid

                # =========================
                # GENERATE EMBEDDING
                # =========================
                embedding = get_batch_embedding([clean_text])  # shape (1, 768)

                # =========================
                # SCALING (IMPORTANT FIX)
                # =========================
                embedding = scaler.transform(embedding)

                embedding_json = json.dumps(
                    embedding[0].tolist()
                )

                # =========================
                # INSERT EMBEDDING
                # =========================
                cursor.execute("""
                    INSERT INTO dataset_embedding
                    (
                        id_dataset,
                        embedding
                    )
                    VALUES
                    (%s, %s)
                """, (
                    id_dataset,
                    embedding_json
                ))

                connection.commit()

            flash("Dataset successfully added", "success")
            return redirect(url_for('dataset'))

        except Exception as e:
            connection.rollback()
            flash(str(e), "danger")

        finally:
            connection.close()

    return render_template('tambah_dataset.html')

@app.route('/edit_dataset/<int:id>', methods=['GET', 'POST'])
def edit_dataset(id):

    if 'user_id' not in session:
        flash("Please log in first.", "warning")
        return redirect(url_for('login'))

    connection = get_db_connection()

    try:
        with connection.cursor() as cursor:

            # =========================
            # GET DATA LAMA
            # =========================
            cursor.execute("""
                SELECT *
                FROM dataset
                WHERE id_dataset = %s
            """, (id,))

            data = cursor.fetchone()

            if not data:
                flash("Data tidak ditemukan", "danger")
                return redirect(url_for('dataset'))

            # =========================
            # HANDLE POST (UPDATE)
            # =========================
            if request.method == 'POST':

                review_id = request.form['review_id']
                content = request.form['content']
                score = request.form['score']
                sentiment = request.form['sentiment']

                translated_text = translator.translate(content)
                clean_text = preprocess_text(translated_text)

                cursor.execute("""
                    SELECT id_dataset
                    FROM dataset
                    WHERE review_id = %s
                    AND id_dataset != %s
                """, (review_id, id))

                existing = cursor.fetchone()

                if existing:
                    flash("Review ID sudah digunakan oleh data lain!", "danger")
                    return redirect(url_for('edit_dataset', id=id))
                # =========================
                # UPDATE DATASET
                # =========================
                cursor.execute("""
                    UPDATE dataset
                    SET review_id = %s,
                        content = %s,
                        score = %s,
                        translated_text = %s,
                        clean_text = %s,
                        sentiment = %s
                    WHERE id_dataset = %s
                """, (
                    review_id,
                    content,
                    score,
                    translated_text,
                    clean_text,
                    sentiment,
                    id
                ))

                # =========================
                # RE-EMBEDDING
                # =========================
                embedding = get_batch_embedding([clean_text])

                embedding = scaler.transform(embedding)

                embedding_json = json.dumps(
                    embedding[0].tolist()
                )

                # =========================
                # UPDATE EMBEDDING
                # =========================
                cursor.execute("""
                    UPDATE dataset_embedding
                    SET embedding = %s
                    WHERE id_dataset = %s
                """, (
                    embedding_json,
                    id
                ))

                connection.commit()

                flash("Dataset successfully updated", "success")
                return redirect(url_for('dataset'))

            return render_template('edit_dataset.html', data=data)

    except Exception as e:
        connection.rollback()
        flash(str(e), "danger")
        return redirect(url_for('dataset'))  # 🔥 INI YANG HILANG

    finally:
        connection.close()

@app.route('/hapus_dataset/<int:id>', methods=['GET', 'POST'])
def hapus_dataset(id):
    if 'user_id' not in session:
        flash(f"Please log in first.", "warning")
        return redirect(url_for('login'))
    connection = get_db_connection()
    if connection is None:
        flash(f'Database connection failed', 'danger')
        return redirect(url_for('dataset'))
    
    try:
        with connection.cursor() as cursor:
            # hapus embedding dulu
            cursor.execute("""
                DELETE FROM dataset_embedding
                WHERE id_dataset = %s
            """, (id,))

            # hapus dataset
            cursor.execute("""
                DELETE FROM dataset
                WHERE id_dataset = %s
            """, (id,))
            connection.commit()
            flash(f'Data successfully deleted', 'success')
    except Exception as e:
        print("Error: {e}")
        flash(f"Error: {e}", 'danger')
    finally:
        connection.close()
    
    return redirect(url_for('dataset'))

@app.route('/import_excel_dataset', methods=['POST'])
def import_excel_dataset():

    if 'user_id' not in session:
        flash('Please log in first.', 'warning')
        return redirect(url_for('login'))

    if 'file' not in request.files:
        flash('No file part', 'danger')
        return redirect(url_for('dataset'))

    file = request.files['file']

    if file.filename == '':
        flash('No selected file', 'danger')
        return redirect(url_for('dataset'))

    if not file.filename.endswith('.csv'):
        flash('Invalid file format', 'danger')
        return redirect(url_for('dataset'))

    try:
        connection = get_db_connection()

        if connection is None:
            flash('Database connection failed', 'danger')
            return redirect(url_for('dataset'))

        # =========================
        # READ CSV SAFER WAY
        # =========================
        import pandas as pd

        df = pd.read_csv(file)

        # =========================
        # DROP NA (INI YANG KAMU MAU)
        # =========================
        df = df.dropna(subset=["clean_text", "sentiment"]).reset_index(drop=True)

        dataset_ids = []

        with connection.cursor() as cursor:

            for _, row in df.iterrows():

                review_id = row["reviewId"]
                content = row["content"]
                score = int(row["score"])
                translated_text = row["translated_text"]
                clean_text = row["clean_text"]
                sentiment = row["sentiment"]

                cursor.execute("""
                    INSERT INTO dataset
                    (
                        review_id,
                        content,
                        score,
                        translated_text,
                        clean_text,
                        sentiment
                    )
                    VALUES (%s,%s,%s,%s,%s,%s)
                """, (
                    review_id,
                    content,
                    score,
                    translated_text,
                    clean_text,
                    sentiment
                ))

        connection.commit()

        flash('Data successfully imported', 'success')

    except Exception as e:
        print(f"Error: {e}")
        flash(f"Error: {e}", 'danger')

    finally:
        connection.close()

    return redirect(url_for('dataset'))

@app.route('/kosongkan_dataset', methods=['GET', 'POST'])
def kosongkan_dataset():
    connection = get_db_connection()
    if connection is None:
        flash(f'Database connection failed', 'danger')
        return redirect(url_for('data_training'))
    
    try:
        with connection.cursor() as cursor:
            cursor.execute("DELETE FROM dataset_embedding")
            cursor.execute("DELETE FROM dataset")
            connection.commit()
            flash(f'Data successfully cleared', 'success')
    except Exception as e:
        print("Error: {e}")
        flash(f"Error: {e}", 'danger')
    finally:
        connection.close()
    
    return redirect(url_for('dataset'))

@app.route('/translateandpreprocessing', methods=['GET', 'POST'])
def translate():

    if 'user_id' not in session:
        flash("Please log in first.", "warning")
        return redirect(url_for('login'))

    # ==========================
    # GET
    # ==========================
    if request.method == 'GET':
        return render_template('translate.html', tables=None, download_link=None)

    # ==========================
    # VALIDASI FILE
    # ==========================
    if 'file' not in request.files:
        flash('No file part', 'danger')
        return render_template('translate.html', tables=None)

    file = request.files['file']

    if file.filename == '':
        flash('No selected file', 'danger')
        return render_template('translate.html', tables=None)

    if not allowed_file(file.filename):
        flash('Invalid file format', 'danger')
        return render_template('translate.html', tables=None)

    try:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # ==========================
        # READ FILE
        # ==========================
        if filename.endswith('.csv'):
            df = pd.read_csv(file_path, encoding='latin1')
        else:
            df = pd.read_excel(file_path)

        # ==========================
        # VALIDASI KOLOM
        # ==========================
        if 'content' not in df.columns:
            flash("Kolom 'content' tidak ditemukan", "danger")
            return render_template('translate.html', tables=None)

        # ==========================
        # TRANSLATE
        # ==========================
        translator = GoogleTranslator(source='auto', target='en')

        df['translated_text'] = df['content'].apply(
            lambda x: translator.translate(str(x))
        )

        # ==========================
        # CLEAN TEXT
        # ==========================
        df['clean_text'] = df['translated_text'].apply(preprocess_text)

        # ==========================
        # AMBIL KOLOM SESUAI UI
        # ==========================
        df_result = df.reindex(columns=[
            'reviewId',
            'content',
            'score',
            'translated_text',
            'clean_text'
        ])

        # ==========================
        # SAVE FILE OUTPUT
        # ==========================
        translated_file = f"translated_{filename}"
        translated_path = os.path.join(app.config['UPLOAD_FOLDER'], translated_file)

        if filename.endswith('.csv'):
            df_result.to_csv(translated_path, index=False, encoding='utf-8')
        else:
            df_result.to_excel(translated_path, index=False)

        # ==========================
        # DOWNLOAD LINK
        # ==========================
        download_link = Markup(
            f'<a href="{url_for("download_labeled_dataset", filename=translated_file)}" class="btn btn-primary">'
            f'Download Result</a>'
        )

        flash('File berhasil diproses', 'success')

        # ==========================
        # RENDER TABLE
        # ==========================
        return render_template(
            'translate.html',
            tables=df_result.to_dict(orient='records'),
            download_link=download_link
        )

    except Exception as e:
        flash(f'Error processing file: {str(e)}', 'danger')
        return render_template('translate.html', tables=None)


@app.route('/pelabelan_dataset', methods=['GET', 'POST'])
def pelabelan_dataset():

    if 'user_id' not in session:
        flash("Please log in first.", "warning")
        return redirect(url_for('login'))

    # =========================
    # GET → tampilkan halaman
    # =========================
    if request.method == 'GET':
        return render_template('pelabelan_dataset.html', tables=None, download_link=None)

    # =========================
    # VALIDASI FILE
    # =========================
    if 'file' not in request.files:
        flash('No file part', 'danger')
        return render_template('pelabelan_dataset.html')

    file = request.files['file']

    if file.filename == '':
        flash('No selected file', 'danger')
        return render_template('pelabelan_dataset.html')

    if not allowed_file(file.filename):
        flash('Invalid file format', 'danger')
        return render_template('pelabelan_dataset.html')

    try:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # =========================
        # READ FILE
        # =========================
        if filename.endswith('.csv'):
            df = pd.read_csv(file_path, encoding='latin1')
        else:
            df = pd.read_excel(file_path)

        # =========================
        # PREPROCESS
        # =========================
        text_column = 'clean_text'
        # =========================
        # SENTIMENT (VADER)
        # =========================
        analyzer = SentimentIntensityAnalyzer()

        labels = []

        for text in df['clean_text']:
            score = analyzer.polarity_scores(str(text))['compound']

            if score >= 0.05:
                labels.append('Positive')
            elif score <= -0.05:
                labels.append('Negative')
            else:
                labels.append('Netral')

        df['sentiment'] = labels

        # =========================
        # SIMPAN FILE RESULT
        # =========================
        result_file = f"labeled_{filename}"
        result_path = os.path.join(app.config['UPLOAD_FOLDER'], result_file)

        df.to_csv(result_path, index=False, encoding='utf-8')

        download_link = Markup(
            f'<a href="{url_for("download_labeled_dataset", filename=result_file)}" class="btn btn-primary">'
            f'Download Result</a>'
        )

        flash("Data berhasil dilabeli", "success")

        return render_template(
            'pelabelan_dataset.html',
            tables=df.to_dict(orient='records'),
            download_link=download_link
        )

    except Exception as e:
        flash(str(e), "danger")
        return render_template('pelabelan_dataset.html', tables=None)

@app.route('/download_labeled_dataset/<filename>')
def download_labeled_dataset(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename, as_attachment=True)

@app.route('/api/training_dataset')
def api_training_dataset():

    draw = request.args.get('draw', type=int, default=1)
    start = request.args.get('start', type=int, default=0)
    length = request.args.get('length', type=int, default=10)

    connection = get_db_connection()

    try:
        with connection.cursor() as cursor:

            cursor.execute("SELECT COUNT(*) AS total FROM dataset")
            total = cursor.fetchone()["total"]

            cursor.execute("""
                SELECT
                    id_dataset,
                    review_id,
                    content,
                    translated_text,
                    clean_text,
                    sentiment
                FROM dataset
                ORDER BY id_dataset
                LIMIT %s OFFSET %s
            """, (length, start))

            rows = cursor.fetchall()

        return jsonify({
            "draw": draw,
            "recordsTotal": total,
            "recordsFiltered": total,
            "data": rows
        })

    finally:
        connection.close()

@app.route('/training', methods=['GET', 'POST'])
def training():

    if 'user_id' not in session:
        flash("Please log in first.", "warning")
        return redirect(url_for('login'))

    connection = get_db_connection()

    try:

        with connection.cursor() as cursor:

            # ==========================================
            # DATASET
            # ==========================================

            cursor.execute("""
                SELECT *
                FROM dataset
                ORDER BY id_dataset
            """)

            dataset = cursor.fetchall()

            # ==========================================
            # LOAD SETTING
            # ==========================================

            cursor.execute("""
                SELECT *
                FROM pengaturan_pengujian
                WHERE id_pengaturanpengujian = 1
            """)

            setting = cursor.fetchone()

            if request.method == 'POST':

                # ==========================================
                # PARAMETER
                # ==========================================

                max_length = int(
                    request.form['max_length']
                )

                batch_size = int(
                    request.form['batch_size']
                )


                var_smoothing = float(
                    request.form['var_smoothing']
                )

                n_estimators = int(
                    request.form['n_estimators']
                )

                learning_rate = float(
                    request.form['learning_rate']
                )

                # ==========================================
                # UPDATE SETTING
                # ==========================================

                cursor.execute("""
                    UPDATE pengaturan_pengujian
                    SET
                        max_length_bert=%s,
                        batch_size_bert=%s,
                        var_smoothing_nb=%s,
                        n_estimators_ab=%s,
                        learning_rate_ab=%s,
                        updated_at=NOW()
                    WHERE id_pengaturanpengujian=1
                """, (
                    max_length,
                    batch_size,
                    var_smoothing,
                    n_estimators,
                    learning_rate
                ))

                connection.commit()

                # ==========================================
                # LOAD DATASET
                # ==========================================

                df = pd.DataFrame(dataset)


                # ==========================================
                # HAPUS EMBEDDING LAMA
                # ==========================================

                cursor.execute("""
                    DELETE FROM dataset_embedding
                """)

                connection.commit()

                # ==========================================
                # LOAD BERT
                # ==========================================

                device = torch.device(
                    "cuda"
                    if torch.cuda.is_available()
                    else "cpu"
                )

                tokenizer = AutoTokenizer.from_pretrained(
                    "bert-base-uncased"
                )

                bert = AutoModel.from_pretrained(
                    "bert-base-uncased"
                )

                bert.to(device)
                bert.eval()

                # ==========================================
                # FIT SCALER
                # ==========================================

                scaler = StandardScaler()

                total = len(df)

                for i in range(
                    0,
                    total,
                    batch_size
                ):

                    batch_texts = (
                        df['clean_text']
                        .iloc[i:i+batch_size]
                        .astype(str)
                        .tolist()
                    )

                    embeddings = get_batch_embedding(
                        batch_texts
                    )

                    scaler.partial_fit(
                        embeddings
                    )

                os.makedirs(
                    'models',
                    exist_ok=True
                )

                joblib.dump(
                    scaler,
                    'models/embedding_scaler.pkl'
                )

                # ==========================================
                # GENERATE EMBEDDING
                # ==========================================

                X = []
                y = []

                insert_sql = """
                INSERT INTO dataset_embedding
                (
                    id_dataset,
                    embedding
                )
                VALUES (%s,%s)
                """

                for i in range(
                    0,
                    total,
                    batch_size
                ):

                    batch_df = df.iloc[
                        i:i+batch_size
                    ]

                    texts = (
                        batch_df['clean_text']
                        .astype(str)
                        .tolist()
                    )

                    embeddings = get_batch_embedding(
                        texts
                    )

                    embeddings = scaler.transform(
                        embeddings
                    )

                    for j, emb in enumerate(
                        embeddings
                    ):

                        X.append(emb)

                        sentiment = (
                            batch_df
                            .iloc[j]
                            ['sentiment']
                        )

                        y.append(
                            sentiment
                        )

                        cursor.execute(
                            insert_sql,
                            (
                                int(
                                    batch_df
                                    .iloc[j]
                                    ['id_dataset']
                                ),
                                json.dumps(
                                    emb.tolist()
                                )
                            )
                        )

                connection.commit()

                # ==========================================
                # PREPARE TRAINING DATA
                # ==========================================

                X = np.array(X)

                encoder = LabelEncoder()

                y_encoded = encoder.fit_transform(
                    y
                )

                X_train, X_test, y_train, y_test = (
                    train_test_split(
                        X,
                        y_encoded,
                        test_size=0.2,
                        random_state=42,
                        stratify=y_encoded
                    )
                )

                # ==========================================
                # MODEL NB
                # ==========================================

                nb_model = GaussianNB(
                    var_smoothing=var_smoothing
                )

                nb_model.fit(
                    X_train,
                    y_train
                )

                # ==========================================
                # ADABOOST + NB
                # ==========================================

                sample_weight = (
                    np.ones(
                        len(X_train)
                    )
                    /
                    len(X_train)
                )

                models = []
                alphas = []

                classes = np.unique(
                    y_train
                )

                for _ in range(
                    n_estimators
                ):

                    clf = GaussianNB(
                        var_smoothing=var_smoothing
                    )

                    clf.fit(
                        X_train,
                        y_train,
                        sample_weight=sample_weight
                    )

                    pred = clf.predict(
                        X_train
                    )

                    incorrect = (
                        pred != y_train
                    )

                    err = np.sum(
                        sample_weight
                        *
                        incorrect
                    )

                    err = np.clip(
                        err,
                        1e-10,
                        0.999999
                    )

                    alpha = (
                        learning_rate
                        *
                        (
                            np.log(
                                (1 - err)
                                /
                                err
                            )
                            +
                            np.log(
                                len(classes)-1
                            )
                        )
                    )

                    sample_weight *= np.exp(
                        alpha *
                        incorrect
                    )

                    sample_weight /= np.sum(
                        sample_weight
                    )

                    models.append(
                        clf
                    )

                    alphas.append(
                        alpha
                    )

                # ==========================================
                # SAVE MODEL
                # ==========================================

                joblib.dump(
                    nb_model,
                    "models/gaussian_nb.pkl"
                )

                joblib.dump(
                    {
                        "models": models,
                        "alphas": alphas,
                        "encoder": encoder
                    },
                    "models/adaboost_nb.pkl"
                )

                flash(
                    "Training berhasil. Embedding dan model berhasil disimpan.",
                    "success"
                )

                return redirect(
                    url_for('training')
                )

    except Exception as e:

        flash(
            str(e),
            "danger"
        )

        dataset = []
        setting = None

    finally:

        connection.close()

    return render_template(
        'training.html',
        username=session['username'],
        dataset=dataset,
        setting=setting
    )

@app.route('/generate_pkl', methods=['GET'])
def generate_pkl():

    if 'user_id' not in session:
        flash("Please log in first.", "warning")
        return redirect(url_for('login'))

    connection = get_db_connection()

    try:

        with connection.cursor() as cursor:

            # ======================================
            # LOAD EMBEDDING
            # ======================================

            cursor.execute("""
                SELECT
                    de.embedding,
                    d.sentiment
                FROM dataset_embedding de
                INNER JOIN dataset d
                    ON de.id_dataset = d.id_dataset
                WHERE de.embedding IS NOT NULL
            """)

            rows = cursor.fetchall()

        if len(rows) == 0:
            flash(
                "Dataset embedding belum tersedia.",
                "warning"
            )
            return redirect(url_for('training'))

        # ======================================
        # DATAFRAME
        # ======================================

        df = pd.DataFrame(rows)

        embeddings = []

        for emb in df["embedding"]:

            embeddings.append(
                np.array(
                    json.loads(emb),
                    dtype=np.float32
                )
            )

        X = np.array(embeddings)

        # ======================================
        # LABEL ENCODER
        # ======================================

        encoder = LabelEncoder()

        y = encoder.fit_transform(
            df["sentiment"]
        )

        # ======================================
        # SPLIT DATA
        # ======================================

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )

        # ======================================
        # DEFAULT PARAMETER
        # ======================================

        var_smoothing = 1e-9
        n_estimators = 10
        learning_rate = 1.0

        # ======================================
        # GAUSSIAN NB
        # ======================================

        nb_model = GaussianNB(
            var_smoothing=var_smoothing
        )

        nb_model.fit(
            X_train,
            y_train
        )

        # ======================================
        # ADABOOST + GNB
        # ======================================

        sample_weight = (
            np.ones(len(X_train))
            / len(X_train)
        )

        models = []
        alphas = []

        classes = np.unique(y_train)

        for _ in range(n_estimators):

            clf = GaussianNB(
                var_smoothing=var_smoothing
            )

            clf.fit(
                X_train,
                y_train,
                sample_weight=sample_weight
            )

            pred = clf.predict(X_train)

            incorrect = (
                pred != y_train
            )

            err = np.sum(
                sample_weight * incorrect
            )

            err = np.clip(
                err,
                1e-10,
                0.999999
            )

            alpha = (
                learning_rate *
                (
                    np.log((1 - err) / err)
                    +
                    np.log(len(classes) - 1)
                )
            )

            sample_weight *= np.exp(
                alpha * incorrect
            )

            sample_weight /= np.sum(
                sample_weight
            )

            models.append(clf)
            alphas.append(alpha)

        # ======================================
        # SAVE MODEL
        # ======================================

        os.makedirs(
            "models",
            exist_ok=True
        )

        joblib.dump(
            nb_model,
            "models/gaussian_nb.pkl"
        )

        joblib.dump(
            {
                "models": models,
                "alphas": alphas,
                "encoder": encoder
            },
            "models/adaboost_nb.pkl"
        )

        flash(
            "Model gaussian_nb.pkl dan adaboost_nb.pkl berhasil dibuat.",
            "success"
        )

    except Exception as e:

        flash(
            str(e),
            "danger"
        )

    finally:

        connection.close()

    return redirect(
        url_for('training')
    )


@app.route('/hasil_pengujian')
def hasil_pengujian():

    if 'user_id' not in session:
        flash("Please log in first.", "warning")
        return redirect(url_for('login'))
    connection = get_db_connection()

    try:

        # =====================================
        # AMBIL DATA DARI DATABASE
        # =====================================

        with connection.cursor() as cursor:

            cursor.execute("""
                SELECT
                    de.embedding,
                    d.sentiment
                FROM dataset_embedding de
                INNER JOIN dataset d
                    ON de.id_dataset = d.id_dataset
                WHERE de.embedding IS NOT NULL
            """)

            rows = cursor.fetchall()

        # =====================================
        # KONVERSI EMBEDDING
        # =====================================

        embeddings = []
        sentiments = []

        for row in rows:

            embedding = row["embedding"]

            if isinstance(embedding, str):
                embedding = json.loads(embedding)

            embeddings.append(
                np.array(
                    embedding,
                    dtype=np.float32
                )
            )

            sentiments.append(
                row["sentiment"]
            )

        X = np.array(embeddings)

        # =====================================
        # LABEL ENCODING
        # =====================================

        encoder = LabelEncoder()

        y = encoder.fit_transform(
            sentiments
        )

        # =====================================
        # SPLIT DATA
        # =====================================

        _, X_test, _, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )

        

        # =====================================
        # LOAD MODEL
        # =====================================

        nb_model = joblib.load(
            "models/gaussian_nb.pkl"
        )

        ada_model = joblib.load(
            "models/adaboost_nb.pkl"
        )

        # =====================================
        # GAUSSIAN NB
        # =====================================

        y_pred_nb = nb_model.predict(
            X_test
        )

        acc_nb = accuracy_score(
            y_test,
            y_pred_nb
        )

        prec_nb = precision_score(
            y_test,
            y_pred_nb,
            average="weighted",
            zero_division=0
        )

        rec_nb = recall_score(
            y_test,
            y_pred_nb,
            average="weighted",
            zero_division=0
        )

        f1_nb = f1_score(
            y_test,
            y_pred_nb,
            average="weighted",
            zero_division=0
        )

        mis_nb = 1 - acc_nb

        cm_nb = confusion_matrix(
            y_test,
            y_pred_nb
        )

        report_nb = classification_report(
            y_test,
            y_pred_nb,
            target_names=encoder.classes_,
            output_dict=True,
            zero_division=0
        )

        # =====================================
        # SIMPAN CONFUSION MATRIX NB
        # =====================================

        plt.figure(figsize=(6, 5))

        sns.heatmap(
            cm_nb,
            annot=True,
            fmt='d',
            cmap='Blues',
            cbar=False,
            square=True,
            linewidths=1,
            xticklabels=encoder.classes_,
            yticklabels=encoder.classes_
        )

        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix Gaussian NB")
        plt.tight_layout()

        nb_image = "cm_nb.png"

        plt.savefig(
            os.path.join("static", nb_image)
        )

        plt.close()


        # =====================================
        # ADABOOST + NB
        # =====================================

        models = ada_model["models"]
        alphas = ada_model["alphas"]

        classes = np.unique(y)

        final_score = np.zeros(
            (
                X_test.shape[0],
                len(classes)
            )
        )

        for alpha, model in zip(
            alphas,
            models
        ):

            pred = model.predict(
                X_test
            )

            for i, c in enumerate(classes):

                final_score[:, i] += (
                    alpha *
                    (pred == c)
                )

        y_pred_ab = np.argmax(
            final_score,
            axis=1
        )

        acc_ab = accuracy_score(
            y_test,
            y_pred_ab
        )

        prec_ab = precision_score(
            y_test,
            y_pred_ab,
            average="weighted",
            zero_division=0
        )

        rec_ab = recall_score(
            y_test,
            y_pred_ab,
            average="weighted",
            zero_division=0
        )

        f1_ab = f1_score(
            y_test,
            y_pred_ab,
            average="weighted",
            zero_division=0
        )

        mis_ab = 1 - acc_ab

        cm_ab = confusion_matrix(
            y_test,
            y_pred_ab
        )

        report_ab = classification_report(
            y_test,
            y_pred_ab,
            target_names=encoder.classes_,
            output_dict=True,
            zero_division=0
        )

        # =====================================
        # SIMPAN CONFUSION MATRIX ADABOOST
        # =====================================

        plt.figure(figsize=(6, 5))

        sns.heatmap(
            cm_ab,
            annot=True,
            fmt='d',
            cmap='Greens',
            cbar=False,
            square=True,
            linewidths=1,
            xticklabels=encoder.classes_,
            yticklabels=encoder.classes_
        )

        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix Gaussian NB + AdaBoost")
        plt.tight_layout()

        ab_image = "cm_ab.png"

        plt.savefig(
            os.path.join("static", ab_image)
        )

        plt.close()

        
        # =====================================
        # VADER (dari dataset_embedding)
        # =====================================
        vader_counts = Counter(sentiments)

        vader_negative = vader_counts.get("Negative", 0)
        vader_neutral  = vader_counts.get("Neutral", 0)
        vader_positive = vader_counts.get("Positive", 0)

        # =====================================
        # NB + ADABOOST (FULL DATASET)
        # =====================================
        final_score_all = np.zeros((X.shape[0], len(classes)))

        for alpha, model in zip(alphas, models):

            pred_all = model.predict(X)

            for i, c in enumerate(classes):
                final_score_all[:, i] += alpha * (pred_all == c)

        y_pred_all = np.argmax(final_score_all, axis=1)
        pred_labels = encoder.inverse_transform(y_pred_all)

        ab_counts = Counter(pred_labels)

        ab_negative = ab_counts.get("Negative", 0)
        ab_neutral  = ab_counts.get("Neutral", 0)
        ab_positive = ab_counts.get("Positive", 0)

        # =====================================
        # DATA CHART
        # =====================================
        labels = ["Negative", "Neutral", "Positive"]

        vader_values = [
            vader_negative,
            vader_neutral,
            vader_positive
        ]

        ab_values = [
            ab_negative,
            ab_neutral,
            ab_positive
        ]

        x = np.arange(len(labels))
        width = 0.35

        # =====================================
        # PLOT BAR CHART
        # =====================================
        plt.figure(figsize=(9,6))

        bars1 = plt.bar(
            x - width/2,
            vader_values,
            width,
            label="VADER Lexicon",
            color="steelblue"
        )

        bars2 = plt.bar(
            x + width/2,
            ab_values,
            width,
            label="NB + AdaBoost",
            color="seagreen"
        )

        # =====================================
        # TAMBAH ANGKA DI ATAS BAR
        # =====================================

        for bar in bars1:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width()/2,
                height,
                f"{int(height)}",
                ha='center',
                va='bottom',
                fontsize=10
            )

        for bar in bars2:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width()/2,
                height,
                f"{int(height)}",
                ha='center',
                va='bottom',
                fontsize=10
            )

        # =====================================
        # FORMATTING CHART
        # =====================================
        plt.xticks(x, labels)
        plt.ylabel("Number of Data")
        plt.xlabel("Sentiment")
        plt.title("Comparison of Sentiment Distribution")
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.3)

        plt.tight_layout()

        # =====================================
        # SAVE IMAGE
        # =====================================
        chart_image = "sentiment_compare.png"

        plt.savefig(
            os.path.join("static", chart_image),
            bbox_inches="tight"
        )

        plt.close()
        # =====================================
        # DEBUG
        # =====================================

        print("Classes :", encoder.classes_)
        print("Accuracy NB :", acc_nb)
        print("Accuracy AdaBoost :", acc_ab)
        
        # =====================================
        # RENDER
        # =====================================

        return render_template(
            "hasil_pengujian.html",

            # Gaussian NB
            acc_nb=round(acc_nb * 100, 2),
            prec_nb=round(prec_nb * 100, 2),
            rec_nb=round(rec_nb * 100, 2),
            f1_nb=round(f1_nb * 100, 2),
            mis_nb=round(mis_nb * 100, 2),

            # AdaBoost
            acc_ab=round(acc_ab * 100, 2),
            prec_ab=round(prec_ab * 100, 2),
            rec_ab=round(rec_ab * 100, 2),
            f1_ab=round(f1_ab * 100, 2),
            mis_ab=round(mis_ab * 100, 2),

            # Confusion Matrix Image
            cm_nb_img=nb_image,
            cm_ab_img=ab_image,

            report_nb=report_nb,
            report_ab=report_ab,
            # 🔥 TAMBAHAN: Grafik Sentimen
            sentiment_chart=chart_image,

            # Label kelas
            labels=list(encoder.classes_)
        )

    except Exception as e:

        flash(
            f"Terjadi kesalahan: {str(e)}",
            "danger"
        )

        print(e)
        raise

        return render_template(
            "hasil_pengujian.html"
        )

    finally:

        connection.close()




@app.route('/logout')
def logout():
    # Hapus session dan arahkan ke halaman login
    session.pop('user_id', None)
    session.pop('username', None)
    flash(f'You have been logged out', 'success')
    return redirect(url_for('login'))

if __name__ == "__main__":
    print("Server start at http://127.0.0.1:8009")
    app.run(
        host="127.0.0.1",
        port=8009,
        debug=True,
        threaded=True
    )

