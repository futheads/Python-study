from douban_top_250 import settings
import mysql.connector

MYSQL_HOSTS = settings.MYSQL_HOSTS
MYSWL_USER = settings.MYSWL_USER
MYSQL_PASSWORD = settings.MYSQL_PASSWORD
MYSQL_PORT = settings.MYSQL_PORT
MYSQL_DB = settings.MYSQL_DB

conn = mysql.connector.connect(user=MYSWL_USER, password=MYSQL_PASSWORD, host=MYSQL_HOSTS, port=MYSQL_PORT, database=MYSQL_DB)
cursor = conn.cursor(buffered=True)

class Sql:

    @classmethod
    def insert_movie(self, title, movieInfo, star, quote):
        sql = "insert into douban_top_250(title, movieInfo, star, quote) values(%(title)s, %(movieInfo)s, %(star)s, %(quote)s)"
        value = {
            "title": title,
            "movieInfo": movieInfo,
            "star": star ,
            "quote": quote
        }
        cursor.execute(sql, value)
        conn.commit()