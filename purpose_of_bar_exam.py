import streamlit as st
import requests
import pdfplumber
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

urls = [
    "https://www.moj.go.jp/content/000006427.pdf"
]

headers = {"User-Agent": "Mozilla/5.0"}

def textParser(text, n=30, bracketDetect=True):
    import unicodedata
    def len_(text):
        cnt = 0
        for t in text:
            if unicodedata.east_asian_width(t) in "FWA":
                cnt += 2
            else:
                cnt += 1
        return cnt

    text = text.splitlines()
    sentences = []
    t = ""
    bra_cnt = ket_cnt = bra_cnt_jp = ket_cnt_jp = 0
    for i in range(len(text)):
        if not bool(re.search(r"\S", text[i])): continue
        if bracketDetect:
            bra_cnt += len(re.findall(r"[\(（]", text[i]))
            ket_cnt += len(re.findall(r"[\)）]", text[i]))
            bra_cnt_jp += len(re.findall(r"[｢「『]", text[i]))
            ket_cnt_jp += len(re.findall(r"[｣」』]", text[i]))
        if i != len(text) - 1:
            if bool(re.fullmatch(r"[A-Z\s]+", text[i])):
                if t != "": sentences.append(t)
                t = ""
                sentences.append(text[i])
            elif bool(
                    re.match(
                        "(\d{1,2}[\.,、．]\s?(\d{1,2}[\.,、．]*)*\s?|I{1,3}V{0,1}X{0,1}[\.,、．]|V{0,1}X{0,1}I{1,3}[\.,、．]|[・•●])+\s",
                        text[i])) or re.match("\d{1,2}．\w", text[i]) or (
                            bool(re.match("[A-Z]", text[i][0]))
                            and abs(len_(text[i]) - len_(text[i + 1])) > n
                            and len_(text[i]) < n):
                if t != "": sentences.append(t)
                t = ""
                sentences.append(text[i])
            elif (
                    text[i][-1] not in ("。", ".", "．") and
                (abs(len_(text[i]) - len_(text[i + 1])) < n or
                 (len_(t + text[i]) > len_(text[i + 1]) and bool(
                     re.search("[。\.．]\s\d|..[。\.．]|.[。\.．]", text[i + 1][-3:])
                     or bool(re.match("[A-Z]", text[i + 1][:1]))))
                 or bool(re.match("\s?[a-z,\)]", text[i + 1]))
                 or bra_cnt > ket_cnt or bra_cnt_jp > ket_cnt_jp)):
                t += text[i]
            else:
                sentences.append(t + text[i])
                t = ""
        else:
            sentences.append(t + text[i])
    return [s.strip() for s in sentences if s.strip()]

# 文単位分割関数
def simple_sentence_split(text):
    sentences = re.split(r'(?<=[。．!?])\s*', text)
    return [s.strip() for s in sentences if s.strip()]

@st.cache_data
def load_all_texts():
    all_texts = []
    merged_text = ""
    for url in urls:
        pdf_path = "purpose_of_bar_exam.pdf"
        try:
            response = requests.get(url, stream=True, headers=headers, timeout=10)
            if response.ok:
                with open(pdf_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
            else:
                continue

            if os.path.getsize(pdf_path) < 1000:
                continue

            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    text = page.extract_text()
                    if text:
                        merged_text += text + "\n"
            # 区切り削除など前処理
            merged_text = re.sub(r'[\s\n]*- \d+ -[\s\n]*', ' ', merged_text)
            merged_text = re.sub(r'[\s\n]*--- ページ \d+ ---[\s\n]*', ' ', merged_text)
            merged_text = re.sub(r' +', ' ', merged_text)
            merged_text = merged_text.strip()
            # 文単位で分割（URLも一緒に保存）
            for sentence in simple_sentence_split(merged_text):
                all_texts.append({"text": sentence, "url": url})
        except Exception:
            continue
    return all_texts

@st.cache_data
def get_vectorizer_and_matrix(all_texts):
    vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(2, 4))
    tfidf_matrix = vectorizer.fit_transform([item["text"] for item in all_texts])
    return vectorizer, tfidf_matrix

# タイトルの代わりにバナー画像を表示（use_container_width推奨）
st.image("tool.jpg", use_container_width=True)

query = st.text_input("気になるキーワード・文章を入力してください:")

if st.button("検索"):
    with st.spinner("PDFを読み込み中です...（初回は時間がかかります）"):
        all_texts = load_all_texts()  # ←ここで初回のみダウンロード＆キャッシュ
        if not all_texts:
            st.error("文章が抽出できませんでした。PDFの内容や分割方法を確認してください。")
            st.stop()
        vectorizer, tfidf_matrix = get_vectorizer_and_matrix(all_texts)

    if query:
        texts_only = [item["text"] for item in all_texts]
        query_vec = vectorizer.transform([query])
        cosine_sim = cosine_similarity(query_vec, tfidf_matrix)
        top_indices = cosine_sim[0].argsort()[::-1]
        st.subheader("関連性の高い文章（上位10件＋後の文章）")
        shown = set()
        count = 0
        for idx in top_indices:
            text = texts_only[idx]
            url = all_texts[idx]["url"]
            if text not in shown:
                after = texts_only[idx + 1] if idx < len(texts_only) - 1 else ""
                st.markdown(f" {text} {after}[元URL]({url})")
                shown.add(text)
                count += 1
            if count >= 10:
                break