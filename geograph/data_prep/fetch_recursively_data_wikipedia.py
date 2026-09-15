import re
import requests
import json


def fetch_titles_from_category_recursive(category, depth, lang='bg', visited=None):
    if visited is None:
        visited = set()

    if category in visited:
        return []

    visited.add(category)
    titles = []
    s = requests.Session()
    url = f"https://{lang}.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "format": "json",
        "list": "categorymembers",
        "cmtitle": category,
        "cmlimit": "max",
    }

    while True:
        response = s.get(url=url, params=params).json()
        members = response.get("query", {}).get("categorymembers", [])
        for member in members:
            ns = member.get("ns")
            if ns == 14:
                print(depth)
                if depth > 2:
                    continue
                subcat_titles = fetch_titles_from_category_recursive(
                    member["title"], depth + 1, lang, visited
                )
                print(subcat_titles)
                titles.extend(subcat_titles)
            elif ns == 0:
                print(member["title"])
                titles.append(member["title"])

        if "continue" in response:
            params["cmcontinue"] = response["continue"]["cmcontinue"]
        else:
            break

    return titles


WIKI_USER_AGENT = "GeoGraph/1.0 (academic research)"


def fetch_article_content(title, lang='bg'):
    url = f"https://{lang}.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "format": "json",
        "titles": title,
        "prop": "extracts",
        "explaintext": True,
        "redirects": 1,
    }
    response = requests.get(url=url, params=params, headers={"User-Agent": WIKI_USER_AGENT}).json()
    page = next(iter(response["query"]["pages"].values()))
    return page.get("extract", "")


def search_article_title(term, lang='bg', prefix_overlap=5):
    url = f"https://{lang}.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "format": "json",
        "list": "search",
        "srsearch": term,
        "srlimit": 3,
    }
    response = requests.get(url=url, params=params, headers={"User-Agent": WIKI_USER_AGENT}).json()
    hits = response.get("query", {}).get("search", [])

    term_lower = term.lower()
    overlap_len = min(prefix_overlap, len(term_lower))
    for hit in hits:
        title_lower = hit["title"].lower()
        if title_lower[:overlap_len] == term_lower[:overlap_len]:
            return hit["title"]
    return None


def clean_text(text):
    cleaned_text = re.sub(r"\(.*?\)", "", text)
    cleaned_text = cleaned_text.replace("\n", " ")
    cleaned_text = cleaned_text.replace('"', "")
    cleaned_text = " ".join(cleaned_text.split())
    return cleaned_text


def save_articles_to_json(titles, filename, lang='bg'):
    articles = []
    for title in titles:
        content = fetch_article_content(title, lang)
        if len(content) > 0:
            content = clean_text(content)
            articles.append({"title": title, "content": content, "language": lang})

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(articles, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    category_bg = "Категория:Селища_в_България"
    titles_bg = fetch_titles_from_category_recursive(category_bg, 0, lang="bg")
    save_articles_to_json(titles_bg, "fetched-wiki-bg-geo-selishta.json", lang="bg")
