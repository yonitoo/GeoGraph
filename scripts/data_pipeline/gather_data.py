import re

import requests
import json


def fetch_titles_from_category(category, lang='bg'):
    s = requests.Session()
    url = f"https://{lang}.wikipedia.org/w/api.php"
    titles = []
    params = {
        "action": "query",
        "format": "json",
        "list": "categorymembers",
        "cmtitle": category,
        "cmlimit": "max"
    }

    while True:
        response = s.get(url=url, params=params).json()
        members = response['query']['categorymembers']
        for member in members:
            titles.append(member['title'])

        if 'continue' in response:
            params['cmcontinue'] = response['continue']['cmcontinue']
        else:
            break

    return titles


def fetch_article_content(title, lang='bg'):
    url = f"https://{lang}.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "format": "json",
        "titles": title,
        "prop": "extracts",
        "explaintext": True,
    }

    response = requests.get(url=url, params=params).json()
    page = next(iter(response['query']['pages'].values()))
    return page.get("extract", "")


def save_articles_to_json(titles, filename, lang='bg'):
    articles = []
    for title in titles:
        content = fetch_article_content(title, lang)
        if len(content) > 0:
            content = clean_text(content)
            articles.append({"title": title, "content": content, "language": lang})

    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(articles, f, ensure_ascii=False, indent=4)


def clean_text(text):
    cleaned_text = re.sub(r'\(.*?\)', '', text)
    cleaned_text = cleaned_text.replace('\n', ' ')
    cleaned_text = cleaned_text.replace('"', '')
    cleaned_text = ' '.join(cleaned_text.split())
    return cleaned_text


category_bg = "Категория:География_на_България"
titles_bg = fetch_titles_from_category(category_bg, 'bg')
save_articles_to_json(titles_bg, 'fetched-wiki-bg-geo-category-no-subcat.json', 'bg')
