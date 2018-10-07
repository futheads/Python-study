import requests
from bs4 import BeautifulSoup

url = "https://www.zhihu.com/explore"
headers = {
    "user-agent": "Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/64.0.3282.186 Safari/537.36"
}

def get_page(url):
    return requests.get(url, headers=headers).text

def get_content(html):
    soup = BeautifulSoup(html, "lxml")
    items = soup.find_all(attrs={"class": "explore-feed feed-item"})
    for item in items:
        question = item.h2.string
        author = item.find(attrs={"class": "author-link"}).string
        pass

        # yield {
        #     "question": item.h2.a.string,
        #     "author": item.find(attrs={"class": "author-link"})
        # }

def save(text):
    with open("explore.txt", "a", encoding="utf-8") as file:
        file.write("\n".join([text["question"], text["author"]]))
    # print("\n".join([text["question"], text["author"], text["answer"]]))

def main(url):
    html = get_page(url)
    for item in get_content(html):
        save(item)

if __name__ == '__main__':
    main(url)