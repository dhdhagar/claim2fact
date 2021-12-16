from urllib.request import Request, urlopen
from bs4 import BeautifulSoup
import pandas as pd
import re
from tqdm import tqdm
import pickle
from urllib.error import HTTPError, URLError
import uuid

out_file = './article_body_titles.pkl'

val_file = './raw_data/val.json'
train_file = './raw_data/train.json'
test_file = './raw_data/test.json'
site_main_tag = {'politifact': 'm-textblock', 'snopes': "single-body card card-body rich-text"} # A bit hacky but works
val_df = pd.read_json(val_file)
train_df = pd.read_json(train_file)
test_df = pd.read_json(test_file)

url_list = train_df['url'].to_list()
url_list.extend(val_df['url'].to_list())
url_list.extend(test_df['url'].to_list())
url_list = list(set(url_list))

failed_list = []
CLEANR = re.compile(r"(\n|\xa0)+")
fetched_data = []

for url in tqdm(url_list):
    req = Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    try:
        webpage = urlopen(req).read()
        content = webpage.decode('utf-8')
        soup = BeautifulSoup(content, 'html.parser')
        for site in site_main_tag.keys():
            if site in url:
                body = soup.find(class_=site_main_tag[site]).get_text()
        body = re.sub(CLEANR, ' ', body).strip()
        title = soup.find('title').get_text()
        fetched_data.append({'body': body, 'title': title, 'url': url})
    except (URLError, HTTPError) as e:
        failed_list.append(url)

print(f"Failed to fetch {len(failed_list)} URLs")

with open(out_file, 'wb') as f:
    pickle.dump(fetched_data, f)


    
