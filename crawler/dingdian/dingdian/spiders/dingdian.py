import scrapy
from scrapy.http import Request
from bs4 import BeautifulSoup

class Myspider(scrapy.Spider):
    name = "dingdian"
    allowed_domains = ["23wx.com"]
    bash_url = "http://www.23wx.com/class/{}_1.html"

    def start_requests(self):
        for i in range(1, 11):
            url = self.bash_url.format(i)
            yield Request(url, self.parse)

    def parse(self, response):
        max_num = BeautifulSoup(response.text, "lxml").find_all("div", class_="pagelink").find_all("a")[-1].get_text()
        bashurl = str(response.url)[:-7]
        for num in range(1, int(max_num) + 1):
            url = bashurl + "_" + str(num)
            yield Request(url, callback=self.get_name)

    def get_name(self, response):
        tds = BeautifulSoup(response.text, "lxml").find_all("tr", bgcolor="#FFFFFF")
        for td in tds:
            novelname = td.find("a").get_text()
            novelurl = td.find("a")["href"]
            yield Request(novelurl, callback=self.get_chapterurl, meta={"name": novelname, "url": novelurl})

    def get_chapterurl(self, response):
        pass