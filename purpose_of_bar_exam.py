import streamlit as st
import requests
import pdfplumber
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import pandas as pd
import pickle

urls = [
    "https://www.moj.go.jp/content/000006427.pdf",
    "https://www.moj.go.jp/content/000006461.pdf",
    "https://www.moj.go.jp/content/000069353.pdf",
    "https://www.moj.go.jp/content/001131108.pdf",
    "https://www.moj.go.jp/content/000095451.pdf",
    "https://www.moj.go.jp/content/000105102.pdf",
    "https://www.moj.go.jp/content/000122708.pdf",
    "https://www.moj.go.jp/content/001130132.pdf",
    "https://www.moj.go.jp/content/001166214.pdf",
    "https://www.moj.go.jp/content/001166215.pdf",
    "https://www.moj.go.jp/content/001166218.pdf",
    "https://www.moj.go.jp/content/001166217.pdf",
    "https://www.moj.go.jp/content/001240101.pdf",
    "https://www.moj.go.jp/content/001240102.pdf",
    "https://www.moj.go.jp/content/001240104.pdf",
    "https://www.moj.go.jp/content/001240105.pdf",
    "https://www.moj.go.jp/content/001273545.pdf",
    "https://www.moj.go.jp/content/001281222.pdf",
    "https://www.moj.go.jp/content/001273548.pdf",
    "https://www.moj.go.jp/content/001273546.pdf",
    "https://www.moj.go.jp/content/001308861.pdf",
    "https://www.moj.go.jp/content/001308862.pdf",
    "https://www.moj.go.jp/content/001308863.pdf",
    "https://www.moj.go.jp/content/001308864.pdf",
    "https://www.moj.go.jp/content/001344496.pdf",
    "https://www.moj.go.jp/content/001344847.pdf",
    "https://www.moj.go.jp/content/001344498.pdf",
    "https://www.moj.go.jp/content/001344499.pdf",
    "https://www.moj.go.jp/content/001358131.pdf",
    "https://www.moj.go.jp/content/001357780.pdf",
    "https://www.moj.go.jp/content/001357781.pdf",
    "https://www.moj.go.jp/content/001357782.pdf",
    "https://www.moj.go.jp/content/001382903.pdf",
    "https://www.moj.go.jp/content/001382904.pdf",
    "https://www.moj.go.jp/content/001382905.pdf",
    "https://www.moj.go.jp/content/001385312.pdf",
    "https://www.moj.go.jp/content/001408691.pdf",
    "https://www.moj.go.jp/content/001408692.pdf",
    "https://www.moj.go.jp/content/001408693.pdf",
    "https://www.moj.go.jp/content/001408694.pdf",
    "https://www.moj.go.jp/content/001429890.pdf",
    "https://www.moj.go.jp/content/001429891.pdf",
    "https://www.moj.go.jp/content/001429892.pdf",
    "https://www.moj.go.jp/content/001429893.pdf"
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

query = st.text_input("気になるキーワード・文章を入力してください。表示に少し時間がかかります。:")

# 検索ボタンが押されたとき
if st.button("検索"):
    with st.spinner("データを読み込み中です..."):
        # all_textsを2分割で読み込み
        df1 = pd.read_csv("all_texts_part1.csv")
        df2 = pd.read_csv("all_texts_part2.csv")
        all_texts1 = df1.to_dict(orient="records")
        all_texts2 = df2.to_dict(orient="records")
        all_texts = all_texts1 + all_texts2

        # all_textsを20分割
        n = len(all_texts) // 20
        all_texts_parts = [all_texts[i*n:(i+1)*n] if i < 19 else all_texts[i*n:] for i in range(20)]

        # vectorizerを読み込み
        with open("vectorizer.pkl", "rb") as f:
            vectorizer = pickle.load(f)

        # クエリベクトルを作成
        if query:
            query_vec = vectorizer.transform([query])

            results = []
            for i in range(20):
                with open(f"tfidf_matrix{i+1}.pkl", "rb") as f:
                    tfidf_matrix = pickle.load(f)
                texts_part = all_texts_parts[i]
                cosine_sim = cosine_similarity(query_vec, tfidf_matrix)
                for idx in range(len(texts_part)):
                    global_idx = i * n + idx
                    score = cosine_sim[0][idx]
                    results.append((global_idx, score))

            # スコア順にまとめて上位10件
            results = sorted(results, key=lambda x: x[1], reverse=True)
            shown = set()
            count = 0
            texts_only = [item["text"] for item in all_texts]
            st.subheader("関連性の高い文章（上位10件＋後の文章）")
            for global_idx, score in results:
                text = texts_only[global_idx]
                url = all_texts[global_idx]["url"]
                if text not in shown:
                    after = texts_only[global_idx + 1] if global_idx < len(texts_only) - 1 else ""
                    st.markdown(f" {text} {after}[元URL]({url})")
                    shown.add(text)
                    count += 1
                if count >= 10:
                    break

# 事前処理用（1回だけ実行）
# import pandas as pd
# import pickle

# all_texts = load_all_texts()  # 必ず定義する
# vectorizer, tfidf_matrix = get_vectorizer_and_matrix(all_texts)

# all_textsをCSVで保存し、2分割
# df = pd.DataFrame(all_texts)
# n = len(df) // 2
# df.iloc[:n].to_csv("all_texts_part1.csv", index=False, encoding="utf-8-sig")
# df.iloc[n:].to_csv("all_texts_part2.csv", index=False, encoding="utf-8-sig")

# tfidf_matrixを20分割で保存
# m = tfidf_matrix.shape[0] // 20
# for i in range(20):
#    start = i * m
#    end = (i + 1) * m if i < 19 else tfidf_matrix.shape[0]
#    with open(f"tfidf_matrix{i+1}.pkl", "wb") as f:
#        pickle.dump(tfidf_matrix[start:end], f)

# vectorizerも保存
# with open("vectorizer.pkl", "wb") as f:
#    pickle.dump(vectorizer, f)