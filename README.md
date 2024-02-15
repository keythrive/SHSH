#### SHSH  Shibata-High-Schools-HandsOn
------

<center>

# 講義ノート (Draft version) <!-- omit in toc -->
## 新潟県立新発田高等学校 <!-- omit in toc -->
#### 研修名： 課題研究のための情報処理 <!-- omit in toc -->
#### 2024/2/19(火) 14:10-16:50 @ 新発田高等学校 <!-- omit in toc -->
#### 講師：　開志専門職大学　情報学部　堀川 桂太郎 <!-- omit in toc -->

</center>

-----
## 目次 <!-- omit in toc -->
- [高校生の課題探求型学習に向けて](#高校生の課題探求型学習に向けて)
  - [課題探求型の重要性](#課題探求型の重要性)
      - [現状に満足せず、妥協しない、成長と挑戦の源泉](#現状に満足せず妥協しない成長と挑戦の源泉)
      - [探求の効能](#探求の効能)
      - [課題の探求方法](#課題の探求方法)
      - [課題探求に必要な資質](#課題探求に必要な資質)
  - [本研修の概要](#本研修の概要)
  - [研修完遂でGetするスキル](#研修完遂でgetするスキル)
  - [演習内容](#演習内容)
  - [事前準備](#事前準備)
  - [予習、自主トレーニング](#予習自主トレーニング)
  - [セッション0.  Pythonでクラシック音楽](#セッション0--pythonでクラシック音楽)
  - [セッション1.  Pythonで動画再生](#セッション1--pythonで動画再生)
  - [セッション2.  現在地の近傍地図](#セッション2--現在地の近傍地図)
  - [セッション3.  球の体積、モンテカルロ法](#セッション3--球の体積モンテカルロ法)
  - [セッション4.  Pythonで 3Dグラフ表示](#セッション4--pythonで-3dグラフ表示)
  - [セッション5.  WordClowd ワードクラウド](#セッション5--wordclowd-ワードクラウド)
      - [源氏物語](#源氏物語)
      - [枕草子](#枕草子)
      - [三国志](#三国志)
      - [三四郎](#三四郎)
      - [月に吠える](#月に吠える)
  - [セッション6.  稀少データは自分で作る（MCMC）](#セッション6--稀少データは自分で作るmcmc)
  - [セッション7.  量子テレポーテーションって何？](#セッション7--量子テレポーテーションって何)
  - [セッション8.  最適化問題を物理法則で解いてもらう](#セッション8--最適化問題を物理法則で解いてもらう)

-----
<div style="page-break-before:always"></div>

# 高校生の課題探求型学習に向けて

## 課題探求型の重要性
#### 現状に満足せず、妥協しない、成長と挑戦の源泉
- 問題解決、よりも、**問題発見**が難しい
- 金鉱採掘、よりも、**金脈発見**が難しい
- 仕事をこなす/探す、よりも、**雇用をつくる**ことは難しい

####  探求の効能
- **能動アウトプット型の自己成長**　：　受動インプット教育よりはるかに高い教育効果
- すべての若者達へ：　　**失敗を恐れず，挑戦する実行力**　
- 優れたアイディアを真似するのは簡単だが，卓越した実行力を模倣することは困難を極める（ルイス・ガースナー）
  
####  課題の探求方法
- 発明好きの発想
- 好奇心と観察
- 多様な視点,　異文化融合

####  課題探求に必要な資質
- 面白い！不思議！と思う感受性
- 困っている人を助ける視点
- 知らないことを知りたい、と思う知識欲
- 普通と違うユニークな視点を受容する広い視野、寛容性

-----
<div style="page-break-before:always"></div>

## 本研修の概要

- 2024年度 **総合的な探求の時間**に役立つ情報処理プラクティス
- 2025年共通テストに、**情報Ⅰ**が追加される背景事情
- データ分析、実践的Pythonプログラミング～初級中級ハンズオン～
- 生成AIを副操縦士としたプログラミング
- インターネット接続されたPC、Webブラウザの対話作業
- 高校生の探究活動へ展開・応用


## 研修完遂でGetするスキル
- 目的・要件⇒デザイン　⇒　PoC確認まで30分
- 「こんなことできるのかな？」　⇒　「**やればできる**」を実感する
- ちょこっとPython: スキマ時間に簡単お手軽プログラミング 
- Holiday Coding:　家族に自慢できるPython使い

-----
<div style="page-break-before:always"></div>

## 演習内容
- Google Colaboratory、Microsoft Bing AI Copilotを使います
- 参加者の皆様のペース配分を見ながら
- 1セッション30分程度でハンズオン × 3～4セッション行います
- 1セッション30分でコーディング＆動作確認まで実施します
- Python初心者はペアプログラミング等follow体制を考えます
- カスタマイズやアレンジは後日、家から実施可能です
- 下記はイメージですが、リストから適宜、ピックアップして演習します

||学習項目|概要、方法|
|-|-|-|
|0|Pythonでクラシック音楽|クラウド環境でPython実行、Webで音楽再生|
|1|Pythonで動画再生|同上、Webで動画再生|
|2|地図と現在地|現在地点の拡大縮小可能な地図を表示|
|3|3D表現|3D空間にランダムな点描画で球を表現、球の体積を試算<br>3Dプリンタの積層方式で図形を立体化|
|4|WordCloud|（国語、英語の先生向け）テキストで頻出する単語を視覚表現|
|5|データ分析・データ生成|神は確率でデータを創造する|
|6|量子ゲート|量子テレポーテーションって何？|
|7|最適化問題の量子的求解|ポートフォリオ最適化,8-Queen,数独パズル|

-----
<div style="page-break-before:always"></div>


## 事前準備

||項目|作業|チェック|
|:-:|-|-|:-:|
|0|Googleアカウント取得|gmailサインアップ・サインイン|✓|
|1|Microsoftアカウント取得|Windows10 ログイン|✓|
|2|WebブラウザEdge起動||✓|
|3|Google Colaboratoryログイン|https://colab.research.google.com/?hl=ja|✓|
|4|新規ノートブック作成|基本操作を試す|✓|
|5|Microsoft Bing AI画面表示|Copilotに質問、基本操作を試す|✓|
|6|基本動作確認（A）|Githubから講義用ファイルをDownloadして、いくつか試す|✓|
|7|基本動作確認（B）|||
|8|Pythonパッケージの事前インストール|（下記参照）||

- 演習のチョイスにもよりますが、可能であれば事前に下記を実行できていると
スムーズに演習に入れると思いますが、講義ノートとともにJupyterファイルを事前にダウンロードできるよう考えております

```
!pip install geopy folium plotly

!apt install aptitude
!aptitude install git make curl xz-utils file -y
!aptitude install mecab libmecab-dev mecab-ipadic-utf8 -y
!apt-get -y install fonts-ipafont-gothic
!pip3 install mecab-python3==0.7

```

## 予習、自主トレーニング
- 必要に応じて、
- Python学習環境　PyWeb集合学習、チュートリアルなど、
- 事前にざっと目を通し、予習やお試しを進めておく：
-     https://pyweb.ayax.jp/PyWeb.html#pywebver

-----
<div style="page-break-before:always"></div>

## セッション0.  Pythonでクラシック音楽

    [ToDo] 
    　クラウド環境でPythonプログラムを作成し実行する. 
    　PCのWebブラウザでクラシック音楽が再生する

1.   クラシック音楽ファイルを用意する：
  (例)  著作権フリーのクラシック音楽サイト: 
    - https://mmt38.info/arrange/morzalt/
    - モーツァルト「トルコ行進曲」をダウンロード
    - toruko.mp3 ⇒　Downloadフォルダへ
2.  Colaboratoryにログイン
3.  新規ノートブック作成
  <img src='./img/new-notebook.png' width=70%>
4.  音楽ファイルをアップロード
  <img src='./img/upload.png' width=70%>
5.  コードセル追加
6.  下記コードを入力、楽曲ファイルのpath  (/content/toruko.mp3)を取得、設定
    <img src='./img/music_path.png' width=70%>

```Python
import IPython
music_path = '/content/toruko.mp3'
print(f'楽曲ファイル:{music_path}')
IPython.display.Audio(music_path)
```
7. セル横の再生ボタンで、実行する
    <img src='./img/music-play.png' width=70%>
8. アレンジ：他の楽曲に変え、自由に楽しむ

-----
<div style="page-break-before:always"></div>

## セッション1.  Pythonで動画再生

    [ToDo] 
    　クラウド環境でPythonプログラムを作成し実行する. 
    　PCのWebブラウザで動画再生する

0.   動画ファイルを用意する：
  (例)  著作権フリーでダウンロード可能な動画ファイル: 
    - https://pixabay.com/ja/videos/
    - 銀河系の動画をダウンロード　https://pixabay.com/ja/videos/%E9%8A%80%E6%B2%B3-%E8%9E%BA%E6%97%8B-%E7%84%A1%E9%99%90%E3%83%AB%E3%83%BC%E3%83%97-%E3%83%AB%E3%83%BC%E3%83%97-107129/
1.  Colaboratoryにログイン
2.  セッション1の続きで、新規セルの追加
3.  動画ファイルをアップロード
4.  動画ファイルのpath  (/content/galaxy.mp3)を取得、設定
5.  下記のコードを入力、

```Python
import io, base64
from IPython.display import HTML
from google.colab import drive

def show_local_mp4_video(file_name, width=320, height=240):
  video_encoded = base64.b64encode(io.open(file_name, 'rb').read()) 
  return HTML(data='''
    <video width="{0}" height="{1}" alt="test" controls>
        <source src="data:video/mp4;base64,{2}" type="video/mp4" />
    </video>'''.format(width, height, video_encoded.decode('ascii'))) 

# drive.mount('/content/gdrive', force_remount=True)
movie = 'galaxy'
video_path = f'/content/{movie}.mp4'
show_local_mp4_video(video_path)
```
6. セル横の再生ボタンで、実行する
    <img src='./img/movie.png' width=80%>
7. アレンジ：他の動画に差し変えていろいろ楽しむ


-----
<div style="page-break-before:always"></div>

## セッション2.  現在地の近傍地図
    [ToDo] クラウド環境でPythonプログラムを実行し、
     - Webブラウザ上に、現在地近傍の地図を表示する：
     - 拡大縮小可能な地図で、簡単な道案内、ナビに利用できる
     - 住所から緯度経度の取得に失敗する場合、予めWebサイトで緯度経度を求めておく

0. 緯度経度、地図表示に必要なPythonパッケージをインストールする
   
``` !pip install geopy folium ```

1.  下記のコードを入力して実行

```python
address = '新潟県新発田市豊町3丁目7-6'
latitude, longitude =  37.939535, 139.33648
# address = "新潟県新潟市中央区米山3-1-53"  
# latitude, longitude = 37.9136367, 139.0572378

from geopy.geocoders import Nominatim
import folium
geolocator = Nominatim(user_agent="my-app")

location = geolocator.geocode(address)
if location is None:
    print("住所が見つかりませんでした。")
else:
    latitude = location.latitude
    longitude = location.longitude
    print(f"住所：{address} 緯度：{latitude} 経度：{longitude}")

niigata_lat = 37.916192
niigata_lon = 139.036413
map = folium.Map(location=[niigata_lat, niigata_lon], zoom_start=12)
folium.Marker([latitude, longitude], popup=address).add_to(map)
map
```
<img src='./img/map-location.png' width=100%>

-----
<div style="page-break-before:always"></div>

## セッション3.  球の体積、モンテカルロ法

    [ToDo] 
    　3D空間にランダムな点描画で球を表現、
    　球の体積を試算：　モンテカルロ法の重要性を確認

5.  下記のコードを入力、
   
```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

r,n = 1,10000

x = np.random.uniform(-r, r, n) 
y = np.random.uniform(-r, r, n) 
z = np.random.uniform(-r, r, n) 
d = np.sqrt(x**2 + y**2 + z**2) 
inside = d < r   # 単位球の内部か否か

volume = 8 * np.sum(inside) / n 
print(f"球の体積の近似値：{volume:.4f}") # 球の体積の近似値：4.2552

fig = plt.figure(figsize=(7, 7)) 
ax = fig.add_subplot(111, projection="3d") 
ax.scatter(x, y, z, c=inside, cmap="coolwarm", alpha=0.5) 

ax.set_title(f"Monte Carlo method for sphere volume (n={n})") 
ax.set_box_aspect((1, 1, 1)) 
plt.show() 
```
<img src='./img/sphere.png' width=60%>


6. グラフィクスを plotlyに変え、視点をマウス操作で変えられる

```python
import numpy as np
import plotly.express as px

r,n = 1,10000

x = np.random.uniform(-r, r, n) 
y = np.random.uniform(-r, r, n) 
z = np.random.uniform(-r, r, n) 
d = np.sqrt(x**2 + y**2 + z**2) 
inside = d < r 

volume = 8 * np.sum(inside) / n 
print(f"球の体積の近似値：　{volume:.4f}")

fig = px.scatter_3d(x=x, y=y, z=z, color=inside, color_continuous_scale=["lightblue", "darkblue"], opacity=0.5) 
fig.update_layout(title=f"Monte Carlo method for sphere volume (n={n})", scene=dict(xaxis_title="x", yaxis_title="y", zaxis_title="z"))
fig.show() 

```
```
  モンテカルロ法試算結果：　球の体積の近似値：4.1912
```
<img src='./img/MC-sphere.png' width=100%>

```python
import math
math.pi *4/3　# 理論値:  4.1887902047863905
```


-----
<div style="page-break-before:always"></div>

## セッション4.  Pythonで 3Dグラフ表示
    [ToDo] 
    　3Dプリンタの積層方式で図形を立体化

```python
import numpy as np
import matplotlib.pyplot as plt

t = np.linspace(0, 2 * np.pi, 100)
x = 16 * np.sin(t) ** 3
y = 13 * np.cos(t) - 5 * np.cos(2 * t) - 2 * np.cos(3 * t) - np.cos(4 * t)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.set_title('Heart Curve 3D')

# Z軸方向に曲線を積層: 3Dプリンタ方式
for z in range(11):
    # 金色からセピア色に変化
    r = 1 - 0.24 * z / 10
    g = 0.84 - 0.15 * z / 10
    b = 0 + 0.56 * z / 10
    # 線の色をグラデーションにする
    ax.plot(x, y, z, linewidth=16, color=(r, g, b))

plt.show()
```
<img src='./img/3d-heart.png' width=50%>

-----

```python
import numpy as np
import plotly.graph_objects as go
t = np.linspace(0, 2 * np.pi, 100)
x = 16 * np.sin(t) ** 3
y = 13 * np.cos(t) - 5 * np.cos(2 * t) - 2 * np.cos(3 * t) - np.cos(4 * t)

fig = go.Figure()
fig.update_layout(title='Heart Curve 3D', scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'))

# Z軸方向に曲線を積層: 3Dプリンタ方式
for z in range(30):
    fig.add_trace(go.Scatter3d(x=x, y=y, z=[z] * len(x), mode='lines', line=dict(colorscale=[[0, 'green'], [0.5, 'red'], [1.0, 'rgb(0, 0, 255)']], cmin=0, cmax=10,width=5)))
    x = x * 0.91 
    y = y * 0.91
fig.show()
```
<img src='./img/3D-plotly-heart.png' width=70%>


-----
<div style="page-break-before:always"></div>

## セッション5.  WordClowd ワードクラウド
    [ToDo] 
     - テキストで頻出する単語を視覚表現する、例えば・・・
     - 源氏物語、枕草子、三四郎、三国志、現代訳論語...の頻出単語

#### 源氏物語
<img src='./img/wc-源氏物語.png' width=50%>

#### 枕草子
<img src='./img/wc-枕草子-wikipedia.png' width=50%>

#### 三国志
<img src='./img/wc-三国志.png' width=50%>

#### 三四郎
<img src='./img/wc-三四郎.png' width=50%>

#### 月に吠える
<img src='./img/wc-月に吠える.png' width=50%>

- 事前準備、各種パッケージインストール
```
!apt install aptitude

!aptitude install git make curl xz-utils file -y
!aptitude install mecab libmecab-dev mecab-ipadic-utf8 -y
!apt-get -y install fonts-ipafont-gothic
!pip3 install mecab-python3==0.7
```

```python
from collections import defaultdict
import re,io
import urllib.request
from zipfile import ZipFile
from wordcloud import WordCloud
import IPython

fpath = '/usr/share/fonts/opentype/ipafont-gothic/ipagp.ttf'
wc = WordCloud(background_color='white',width=640,height=480,font_path=fpath)

import MeCab
sentense = '花の色は移りにけりないたずらにわがみよにふる眺めせしまに'
m = MeCab.Tagger()
m.parse(sentense)

def show_top(N,txt):
    dic = defaultdict(int)
    m = MeCab.Tagger()
    node = m.parseToNode(txt)
    while node:
        key = node.surface
        a = node.feature.split(',')
        if a[0]==u'名詞' and a[1]==u'一般' and key != '':
            dic[key] += 1
        node = node.next
    for k,v in sorted(dic.items(),key=lambda x:-x[1])[0:N]:
        print(k + ':' + str(v))

def get_words(txt):
    m = MeCab.Tagger()
    node = m.parseToNode(txt)
    w = ''
    while node:
        a = node.feature.split(',')
        if a[0]==u'名詞' and a[1]==u'一般':
            w += node.surface + ' '
        node = node.next
    return w

def load_from_url(url):
    data = urllib.request.urlopen(url).read()
    zipdata = ZipFile(io.BytesIO(data))
    filename = zipdata.namelist()[0]
    txt = zipdata.read(filename).decode('shift-jis')
    txt = re.sub(r' [.*?]','',txt)
    txt = re.sub(r' <.*?>','',txt)
    return txt

def process(URLs):
    for u in URLs:
        txt = load_from_url(u)
        title = txt.split()[0]
        print(f'TITLE={title}\n')   # print(f'Text={txt}\n')
        words = get_words(txt)
        wc.generate(words)
        fig = f'wc-{title}.png'
        wc.to_file(fig)
        IPython.display.Image(fig)
URLs = [
    'https://www.aozora.gr.jp/cards/000052/files/5016_ruby_9746.zip',  # 紫式部　源氏物語
    'https://www.aozora.gr.jp/cards/001562/files/52409_ruby_51058.zip', # 三国志　吉川英治
    'https://www.aozora.gr.jp/cards/001097/files/43785_ruby_58793.zip', # 現代訳論語
    'https://www.aozora.gr.jp/cards/000148/files/794_ruby_4237.zip', # 三四郎
    'https://www.aozora.gr.jp/cards/000067/files/859_ruby_21655.zip',  # 月に吠える　萩原朔太郎
]
process(URLs)

from bs4 import BeautifulSoup
import requests
def get_html(url):
    response = requests.get(url)
    if response.status_code == 200:
        return response.text
    else:
        return f"Error: {response.status_code}"
def extract_japanese_text(html):
    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text()
    text = "".join([c for c in text if ord(c) >= 0x3000 and ord(c) <= 0x9FFF])
    return text

title = 'Python-wikipedia'
url = 'https://ja.wikipedia.org/wiki/Python'
html = get_html(url)
text = extract_japanese_text(html)
print(text)

title = '枕草子-wikipedia'
url = 'https://bungobungo.jp/text/hakbn/'
html = get_html(url)
text = extract_japanese_text(html)
print(text)

words = get_words(text)
wc.generate(words)
fig = f'wc-{title}.png'
wc.to_file(fig)
IPython.display.Image(fig)

url = 'https://ja.wikipedia.org/wiki/%E6%BA%90%E6%B0%8F%E7%89%A9%E8%AA%9E'

title = '源氏物語'
html = get_html(url)
text = extract_japanese_text(html)
print(text)

words = get_words(text)
wc.generate(words)
fig = f'wc-{title}.png'
wc.to_file(fig)
IPython.display.Image(fig)
```


-----
<div style="page-break-before:always"></div>

## セッション6.  稀少データは自分で作る（MCMC）

- データがなければ、データ分析は始まらない
- 稀少データは、集めることさえ難しい
- エントロピーに逆行し、創生されるデータには偏りがある
- 神のみぞしる確率分布
- 集めることが難しいデータは、神頼みのサンプリング


-----
<div style="page-break-before:always"></div>

## セッション7.  量子テレポーテーションって何？

```python
!pip install qiskit qiskit-aer
!pip install qiskit[visualization]
!pip install --upgrade pylatexenc

# 　一回だけノートブックのカーネルを再起動 　（１回限り）
import os
os.kill(os.getpid(), 9)
```

```python
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, execute, Aer
from qiskit.visualization import plot_histogram, plot_bloch_multivector, plot_state_qsphere

# 3量子ビット
qr = QuantumRegister (3)
# 2古典ビット
crz = ClassicalRegister (1)
crx = ClassicalRegister (1)
qc = QuantumCircuit (qr, crz, crx)

# アリスが送りたい量子ビットを初期化 （|1>とする）
qc.x (0)        # |0> ビット反転
qc.barrier ()

# アリスとボブが共有するベル状態ペア
# (量子エンタングルメント＝量子もつれ)
qc.h (1)        #  アダマールゲート:   量子重ね合わせ
qc.cx (1, 2)    #  CNOT ：　制御NOTE
qc.barrier ()

# アリスが自分の量子ビットに操作を施す
qc.cx (0, 1)     #  CNOT ：　制御NOTE
qc.h (0)         #  アダマールゲート:   量子重ね合わせ
qc.barrier ()

# アリスが自分の量子ビットを測定
# 結果を古典レジスタに送る
qc.measure (0, 0)
qc.measure (1, 1)
qc.barrier ()

# ボブが古典レジスタの値に応じて自分の量子ビットを操作 (if文)
qc.x (2).c_if (crx, 1)  # アリスからの古典通信が01 → Xゲートでビット反転
qc.z (2).c_if (crz, 1)  # アリスからの古典通信が10 → Zゲートで位相反転
# アリスからの古典通信が11 → ZゲートZゲート（＝＝Yゲート）でビット位相反転
# アリスからの古典通信が00 → 何もしなくて，そのままのボブの量子ビットでよい

qc.draw (output='mpl')
```

<img src='./img/Q-gate.png' width=80%>

```python
backend = Aer.get_backend('statevector_simulator')
result = execute(qc, backend).result()
statevector = result.get_statevector()

plot_bloch_multivector(statevector)
plot_state_qsphere(statevector)
```
<img src='./img/Q-bloch.png' width=80%>


-----
<div style="page-break-before:always"></div>

## セッション8.  最適化問題を物理法則で解いてもらう

-----
<div style="page-break-before:always"></div>
