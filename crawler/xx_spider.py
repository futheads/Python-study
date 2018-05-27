import requests
from  lxml import etree

headers = {'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/52.0.2743.116 Safari/537.36 Edge/15.15063'}

def get_page(url):
    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            html_doc = str(response.content, 'utf-8')
            return html_doc
    except requests.ConnectionError:
        return None

def get_urls(html):
    selector = etree.HTML(html)
    contents = selector.xpath("//ul/li")
    for each in contents:
        yield {
            "url": each.xpath("a/@href")[0],
            "name": each.xpath("a/@title")[0]
        }

def get_details(url):
    url = "https://www.3344ez.com" + url
    detail_html = get_page(url)
    detail_selector = etree.HTML(detail_html)
    details = detail_selector.xpath('//div[@class="news"]/text()')
    # return {
    #     "url": url,
    #     "by_the_time": details[2],
    #     "actor": details[4],
    #     "size": details[12],
    #     "format": details[14],
    #     "is_censored": details[15],
    #     "thunder": details[17]
    # }
    return (url, details)

def main(page_num):
    html = get_page("https://www.3344ez.com/htm/downlist4/{0}.htm".format(page_num))
    for item in get_urls(html):
        print(item.get("name"), get_details(item.get("url")))

if __name__ == '__main__':
    main(1)
