from tqdm import tqdm
from selenium import webdriver
from selenium.common.exceptions import TimeoutException, WebDriverException
import pandas as pd
import time

position = ["北京","天津","上海","重庆","河北","山西","辽宁","吉林","黑龙江","江苏","浙江","安徽","福建","江西","山东","河南","湖北","湖南","广东","海南","四川","贵州","云南","陕西","甘肃","青海","台湾","内蒙古","广西","西藏","宁夏","新疆","香港","澳门"]
# position = ["北京","天津","上海"]

name, level, hot, address, num = [], [], [], [], []

def get_page(key, page):
    try:
        options = webdriver.ChromeOptions()
        options.add_argument("--headless")
        options.add_experimental_option("excludeSwitches", ["ignore-certificate-errors"])
        driver = webdriver.Chrome(executable_path="D:/program/webdriver/chromedriver.exe", chrome_options=options)
        time.sleep(1)

        url = "http://piao.qunar.com/ticket/list.htm?keyword=" + str(key) + "&region=&from=mpl_search_suggest&page=" + str(page)
        driver.get(url)
        infos = driver.find_elements_by_class_name("sight_item")
        for info in infos:
            name.append(info.find_element_by_class_name("name").text)
            try:
                level.append(info.find_element_by_class_name("level").text)
            except:
                level.append("")
            hot.append(info.find_element_by_class_name("product_star_level").text[3:])
            address.append(info.find_element_by_class_name("area").text)
            try:
                num.append(info.find_element_by_class_name("hot_num").text)
            except:
                num.append(0)
        driver.quit()
    except TimeoutException or WebDriverException:
        print("爬取失败")

if __name__ == '__main__':
    for key in tqdm(position):
        print("正在爬取{}".format(key))
        # 获取前14页
        for page in range(1):
            print("正在获取第{}页".format(page))
            get_page(key, page)
sight = {"name": name, "level": level, "hot": hot, "address": address, "num": num}
sight = pd.DataFrame(sight)
sight.to_csv("sight.csv", encoding="utf-8")