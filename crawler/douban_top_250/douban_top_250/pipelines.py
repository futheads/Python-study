# -*- coding: utf-8 -*-

# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://doc.scrapy.org/en/latest/topics/item-pipeline.html
from douban_top_250.items import Item
from douban_top_250.sql import Sql

class DoubanTop250Pipeline(object):

    def process_item(self, item, spider):
        if isinstance(item, Item):
            Sql.insert_movie(item["title"], item["movieInfo"], item["star"], item["quote"])
        # return item
