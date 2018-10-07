#!/usr/bin/python
# -*- coding: utf-8 -*-

from selenium import webdriver
import time
# from bs4 import BeautifulSoup
# from pyvirtualdisplay import Display
# from selenium.common.exceptions import NoSuchElementException
from selenium.common.exceptions import StaleElementReferenceException, NoSuchElementException, NoSuchWindowException

from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
import numpy as np
# import urllib2
# import urllib
# import pdfkit
# import requests
# import os
# import json
import warnings
# import csv
# from datetime import datetime, timedelta
import logging

# generate pdf
# from reportlab.lib.enums import TA_JUSTIFY
# from reportlab.lib.pagesizes import legal
# from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
# from reportlab.platypus import Table, TableStyle
# from reportlab.lib import colors
# from reportlab.platypus import (Flowable, Paragraph, SimpleDocTemplate, Spacer)
# from reportlab.pdfbase import pdfmetrics
# from reportlab.pdfbase.ttfonts import TTFont
# from reportlab.lib.colors import (
#     black,
#     purple,
#     white,
#     yellow,
#     red
# )

logging.basicConfig()
logger = logging.getLogger("LloydsCrawler")
logger.setLevel(logging.DEBUG)

warnings.filterwarnings("ignore")


class lloyds_driver():
    def __init__(self, userid, password):
        self.account = userid
        self.passwd = password
        # self.display = Display(visible=0, size=(1920, 1080))
        # self.display.start()
        user_agents = [
            'Mozilla/5.0 (Windows; U; Windows NT 5.1; it; rv:1.8.1.11) Gecko/20071127 Firefox/2.0.0.11',
            'Opera/9.25 (Windows NT 5.1; U; en)',
            'Mozilla/4.0 (compatible; MSIE 6.0; Windows NT 5.1; SV1;.NET CLR 1.1.4322; .NET CLR2.0.50727)',
            'Mozilla/5.0 (compatible; Konqueror/3.5; Linux) KHTML/3.5.5(like Gecko) (Kubuntu)',
            'Mozilla/5.0 (X11; U; Linux i686; en-US; rv:1.8.0.12) Gecko/20070731 Ubuntu/dapper-security Firefox/1.5.0.12',
            'Lynx/2.8.5rel.1 libwww-FM/2.14 SSL-MM/1.4.1 GNUTLS/1.2.9',
            "Mozilla/5.0 (X11; Linux i686) AppleWebKit/535.7 (KHTML, like Gecko) Ubuntu/11.04 Chromium/16.0.912.77 Chrome/16.0.912.77 Safari/535.7",
            "Mozilla/5.0 (X11; Ubuntu; Linux i686; rv:10.0) Gecko/20100101 Firefox/10.0 ",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/56.0.2924.87 Safari/537.36"
        ]
        agent = np.random.choice(user_agents)
        # profile = webdriver.Firefox()
        # profile.set_preference("general.useragent.override",
        #                        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36")
        # profile.accept_untrusted_certs = True

        # self.driver = webdriver.Firefox(profile)

        self.driver = webdriver.Firefox()
        self.pdf_path = './static/lloyds/pdf_file_prev/'

        self.homepage_url = 'https://www.lloydslistintelligence.com/'
        self.success_login_url = 'https://www.lloydslistintelligence.com/vessels/'
        self.driver.get(self.homepage_url)
        time.sleep(3)
        self.driver.find_element_by_id('Login').click()
        logger.info("initial a lloyds driver ...")
        logger.info("Start Login ...")
        # wait till login page loaded successfully
        WebDriverWait(self.driver, 30).until(
            EC.visibility_of_element_located((By.ID, 'thePage:siteTemplate:j_id27:username')))

        self.driver.find_element_by_id('thePage:siteTemplate:j_id27:username').send_keys(self.account)
        self.driver.find_element_by_id('thePage:siteTemplate:j_id27:password').send_keys(self.passwd)

        self.driver.find_element_by_name('thePage:siteTemplate:j_id27:j_id73').click()
        time.sleep(3)
        logger.info("++++++++++++++++++++++++++")

    def close(self):
        self.driver.close()
        self.driver.quit()
        # self.display.stop()

    def login_test(self):
        try:
            self.driver.get(self.success_login_url)
            time.sleep(3)

            WebDriverWait(self.driver, 30).until(
                EC.visibility_of_element_located((By.CLASS_NAME, 'lli-searchform__input')))
            self.driver.find_element_by_class_name('lli-searchform__input')
            logger.info("Already logged in")  # Need password input
            return True
        except Exception as e:  # No password necessary
            logger.warning("Did not yet login")
            return False

    def get_result_batch(self, search_word, search_time, num):

        logger.info("--------------开始执行get_result_batch--------------")
        results = []

        for i in range(num):
            logger.info("--------------开始执行搜索--------------")
            search_word = search_word[i]
            search_time = search_time[i]

            logger.info("--------------开始访问vessels--------------------")
            self.driver.get("https://www.lloydslistintelligence.com/vessels")
            logger.info("--------------访问vessels成功--------------------")


            #             self.driver.execute_script("window.open('https://www.lloydslistintelligence.com/vessels/?term=golden%20daisy')")
            time.sleep(10)
            self.driver.save_screenshot("takephoto1.png")

            WebDriverWait(self.driver, 30).until(
                EC.presence_of_element_located((By.CLASS_NAME, 'lli-searchform__input')))

            box = self.driver.find_element_by_class_name('lli-searchform__input')
            box.send_keys(search_word)

            time.sleep(10)
            box_btn = self.driver.find_element_by_class_name('lli-btn-icon').click()

            time.sleep(10)


            # wait until search text in shown in the page(loaded fully)
            while True:
                try:
                    box_value = box.get_attribute("value")
                    break
                except StaleElementReferenceException:
                    pass
            while box_value != search_word:
                time.sleep(2)

        return results


if __name__ == "__main__":
    username = "yshao@bocusa.com"
    password = "boc12345"

    lz = lloyds_driver(username, password)
    lz.get_result_batch(["golden daisy"], ["01-15-2018"], 1)
    lz.close()