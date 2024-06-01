import yfinance as yf
import streamlit as st
import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import sent_tokenize
import string #untuk casefolding
from nltk.tokenize import word_tokenize
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory, StopWordRemover, ArrayDictionary
import Sastrawi.Stemmer.StemmerFactory as sf
stemmer = sf.StemmerFactory().create_stemmer()
import gensim
from gensim import corpora
from gensim.models import LdaModel
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Segmentasi kalimat
def segmentasi(kalimat):
    teks=kalimat.replace('\n',' ')
    teks1=teks.replace('\r',' ')
    segmens = sent_tokenize(teks1)
    return segmens

# Casefolding
def case_folding(segmens):
    case_foldings=[]
    for sent in segmens:
        lower_case=sent.lower()
        hasil = lower_case.translate(str.maketrans("","",string.punctuation))
        hasil = hasil.strip()
        hasil = re.sub(r"\d+", "", hasil)
        case_foldings.append(hasil)
    return case_foldings

# Tokenisasi
def tokenize(casefoldings):
    tokens=[]
    for sent in casefoldings:
        tokens.append(word_tokenize(sent))
    return tokens

# Stopword
def stopwordRemoval(tokens):
    stopword_factory = StopWordRemoverFactory()
    more_stopwords = ["kata", "aku", "jadi", "fa", "satu", "dua", "tiga", "empat", "lima", "enam", "tujuh", "delapan", "sembilan", "sepuluh", "tbk", "dengan", "pan", "ltd", "ia", "bahwa", "lalu", "anz", "belum", "s", "oleh", "kan", "ujar", "kalau", "ingin", "ada", "tersebut", "ton", "kutip", "tetap", "meski", "se", "juta", "unit", "yoga", "tagar", "sedia", "cnn", "kid", "gil", "nva", "gia", "fc", "wib", "ka", "pal", "sangat", "top", "gia", "lalu", "pukul", "apa", "niro", "the", "rer", "de", "pak", "suaracom", "ags", "apabila", "rimanews", "paja", "totok", "bukan", "lebih", "kian", "tak", "pas", "wis", "kerap", "mungkin", "benarbenar", "sahsah", "b"]
    data = stopword_factory.get_stop_words() + more_stopwords   
    dictionary = ArrayDictionary(data)
    stopword = StopWordRemover(dictionary)

    stopword_sentence = []
    for sents in tokens:
        stopword_sentence.append([stopword.remove(word) for word in sents])
    return stopword_sentence

#Stemming
def stemming(stopwords):
    stemmed_sentence = []
    for sents in stopwords:
        stemmed_sentence.append([stemmer.stem(word) for word in sents])
    return stemmed_sentence


#cleaning
# Menghapus String Kosong
def deleteEmptyString(stemmed):
    final_preprocessing_sentence = []
    for sents in stemmed:
        final_preprocessing_sentence.append([word for word in sents if word != ''])
    return final_preprocessing_sentence

#Menggabungkan kata ke kalimat
def finalSentence(finalprocessing):
    final_sentence = []

    for sents in finalprocessing:
        sent = ""
        for word in sents:
            sent += word + " "
        final_sentence.append(sent)
    return final_sentence

#LDA
def getQueryLDA(document):
  # Menghitung tf-idf
  vectorizer = TfidfVectorizer()
  X = vectorizer.fit_transform(document)
  feature_names = vectorizer.get_feature_names_out()
  # Konversi matriks tfidf ke dalam format Gensim
  corpus = gensim.matutils.Sparse2Corpus(X, documents_columns=False)
  # Membuat kamus (dictionary) Gensim
  id2word = corpora.Dictionary.from_corpus(corpus, id2word=dict((i, s) for i, s in enumerate(feature_names))) 
  # Nilai eta 1 / jumlah vocab
  eta_value = 1/len(feature_names)
  # Penerapan LDA model
  lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=2,
                                           alpha=0.5,
                                           eta=eta_value,
                                           per_word_topics=True,
                                           iterations=500
                                           )
  doc_topics = lda_model.get_document_topics(corpus[0])
  if lda_model.per_word_topics:
    # Jika per_word_topics adalah True
    # Dokumen yang dikembalikan adalah daftar tupel (topic_id, probability)
    row = sorted(doc_topics, key=lambda x: x[1], reverse=True)
  else:
    # Jika per_word_topics adalah False
    # Dokumen yang dikembalikan adalah probabilitas topik tunggal
    row = [(0, doc_topics)]  # Dibuat sebagai tupel agar bisa diurutkan
  row = sorted(row, key=lambda x: x[1], reverse=True)
  wp = lda_model.show_topic(row[0][0])
  kata_kunci_topik = ", ".join([word for word, prop in wp])

  return kata_kunci_topik


#MMR
def MMR(Si, query, Sj, lamda):
    
    sentences = [Si] + [query] + Sj
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(sentences)

    # Hitung cosine similarity antara Si and query
    sim1 = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0, 0]
    left_operation = lamda * sim1

    # Hitung cosine similarity antara Si and tiap kalimat di Sj
    sim2_values = cosine_similarity(tfidf_matrix[0], tfidf_matrix[2:])
  
    filtered_values = sim2_values[sim2_values < 1]
    
   
    right_operation = (1 - lamda) * np.max(filtered_values)
    

    # hitung MMR score
    mmr_score = left_operation - right_operation

    return mmr_score


def makeSummary(sentences, query, summary_length, lamda):
    # querySent = ' '.join(query)
    rangkuman = []
    sum_len = len(rangkuman)
    
    while sum_len < summary_length :
        MMRval = {}
        #query = kata kunci LDA
        #sent = kandidat kalimat yang dibuat rangkuman (yg sudah di preprocesiing)
        #rangkuman = hasil dari ARGMAX MMR
        #MMRval = kalimat dan skor MMRnya
        for sent in sentences:
            MMRval[sent] = MMR(sent, query, sentences, lamda)

        maxmmr = max(MMRval, key=MMRval.get)
        rangkuman.append(maxmmr)
        sentences.remove(maxmmr)
        sum_len = len(rangkuman)

    return rangkuman

def gabung(rangkuman, dictionary):
    summ_arr = []
    for index, summ in enumerate(rangkuman) :
        summ_arr.append(dictionary[summ])

    summary = ' '.join(summ_arr)

    return summary


def preprocessing(teks):
    segmens=segmentasi(teks)
    case_foldings=case_folding(segmens)
    tokens=tokenize(case_foldings)
    stopwords=stopwordRemoval(tokens)
    stemmed=stemming(stopwords)
    secondstopwords=stopwordRemoval(stemmed)
    clean=deleteEmptyString(secondstopwords)
    final=finalSentence(clean)
    return final

def getQuery(teks):
    kata_kunci = getQueryLDA(teks)
    return kata_kunci

def summarize(teks, teksprep, kata_kunci, compresrate):
    beforePreprocessing = segmentasi(teks)
    afterPrepocessing = teksprep

    #mengambil teks berita dari kolom final_sentence, yg sudah preprocessing valuenya
    teks = []
    for key, value in enumerate(afterPrepocessing):
        temp = value
        teks.append(temp)

    #mengambil teks berita dari kolom segmen, yg hanya melalui proses Segmentasi
    teksBef = []
    for key1, value1 in enumerate(beforePreprocessing):
        temp1 = value1
        teksBef.append(temp1)

    #teks1 adalah kalimat yg sudah di preprocessing [teks index ke-0][teks index ke-1] dst
    teks1 = teks.copy()
    teks2 = teksBef.copy()

    #membuat dictionary dengan key kalimat yang sudah di preprocessing dan valuenya kalimat yang hanya melalui segmentasi
    dictionary = dict(zip(teks1, teks2))
    
    #query
    queryVar=kata_kunci

    jumlahRangkuman = compresrate * len(teks) / 100

    #mendapatkan n kalimat terbaik dengan MMR
    summaryMMR = makeSummary(teks1, queryVar, jumlahRangkuman, 1)
    
    #rangkuman hasil MMR dengan query berdasarkan kata kunci LDA  
    hasilRingkasan = gabung(summaryMMR, dictionary) 
    return hasilRingkasan



####################------UI------####################
st.title("Summarization Web App")

st.write("""Pahami artikel secara lebih ringkas dan mudah!""")

txt = st.text_area("**Text berita yang ingin diringkas:**",height=250)

genre = st.radio("**Berapa tingkat kompresi yang anda inginkan?**",["30%", "50%"])

compression = 0
if genre == "30%":
    compression = 30
else:
    compression = 50

kata_kunci = ""
ringkasan = []
col1, col2 = st.columns([1,1])
ringkas= col2.button("Ringkas", type="primary")

if ringkas:
    document = preprocessing(txt)
    kata_kunci=getQuery(document)
    st.header("**Kata Kunci:**")
    st.write(kata_kunci)
    ringkasan=summarize(txt, document, kata_kunci, compression)
    st.header("**Ringkasan:**")
    st.write(ringkasan)


# MEMNAMPILKAN DATA UJI COBA
# Fungsi untuk membaca dan memproses data dari file JSON
def load_data(json_path):
    data = pd.read_json(json_path, lines=True)
    data['nomor dokumen'] = range(1, len(data) + 1)
    cols = ['nomor dokumen'] + [col for col in data if col != 'nomor dokumen']
    return data[cols]

# Path ke file JSON
json_paths = [
    'query_results.json',  # dataquery
    'statistik_skenario1-50_results.json',  # data1
    'statistik_skenario1-30_results.json',  # data2
    'statistik_skenario2-50_results.json',  # data3
    'statistik_skenario2-30_results.json',  # data4
    'statistik_skenario3-50_results.json',  # data5
    'statistik_skenario3-30_results.json'   # data6
]

# Memuat semua data
data_list = [load_data(path) for path in json_paths]

# Fungsi untuk menampilkan judul dan dataframe
def display_data(title, data):
    st.write(title)
    st.dataframe(data)

# Judul section
st.header('Hasil Uji Coba')
st.write('**Berikut adalah hasil uji coba peringkasan teks berita menggunakan kata kunci dari Latent Dirichlet Allocation dan metode peringkasan Maximum Marginal Relevance dengan dataset indosum sebanyak 50 artikel.**')



# Judul dan DataFrame
titles = [
    "Hasil Ekstraksi kata kunci dari LDA",
    "Hasil Uji coba peringkasan Lambda 0 dan compression rate 50%",
    "Hasil Uji coba peringkasan Lambda 0 dan compression rate 30%",
    "Hasil Uji coba peringkasan Lambda 0.7 dan compression rate 50%",
    "Hasil Uji coba peringkasan Lambda 0.7 dan compression rate 30%",
    "Hasil Uji coba peringkasan Lambda 1 dan compression rate 50%",
    "Hasil Uji coba peringkasan Lambda 1 dan compression rate 30%"
]

# Menampilkan semua data
for title, data in zip(titles, data_list):
    display_data(title, data)