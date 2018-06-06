import pymysql

class MySQLUtil:

    def __init__(self):
        self.db = pymysql.connect(host="localhost", user="root", password="futhead", port=3306, db='spiders')
        self.cursor = self.db.cursor()
        self.table = "students"

    def connect_db(self):
        self.cursor.execute("select version()")
        data = self.cursor.fetchone()
        print("Database version: {0}".format(data))
        self.cursor.execute("create database spiders default character set utf8")

    def create_table(self, table_name):
        sql = "CREATE TABLE IF NOT EXISTS {0} (id VARCHAR(255) NOT NULL, name VARCHAR(255) NOT NULL, age INT NOT NULL, PRIMARY KEY (id))"
        self.cursor.execute(sql.format(table_name))

    def insert(self, id, user, age):
        sql = "insert into students(id, name, age) values(%s, %s, %s)"
        try:
            self.cursor.execute(sql, (id, user, age))
            self.db.commit()
        except:
            self.db.rollback()

    def common_insert(self):
        data = {
            "id": "123456",
            "name": "mohan",
            "age": 18
        }
        keys = ", ".join(data.keys())
        values = ", ".join(["%s"] * len(data))
        sql = "insert into {table}({keys}) values({values})".format(table=table, keys=keys, values=values)
        try:
            if self.cursor.execute(sql, tuple(data.values())):
                print("Successful")
                self.db.commit()
        except:
            print("Faield")
            self.db.rollback()

    def update(self):
        sql = "update students set age = %s where name = %s"
        try:
            self.cursor.execute(sql, (30, 'futhead'))
            self.db.commit()
        except Exception as e:
            print(e)
            self.db.rollback()

    def upsert(self):
        data = {
            'id': '12345',
            'name': 'Bob',
            'age': 21
        }
        keys = ", ".join(data.keys())
        values = ", ".join(['%s'] * len(data))
        sql = "insert into {table}({keys}) values({values}) on duplicate key update "\
            .format(table=self.table, keys=keys, values=values)
        update = ", ".join(['{key} = %s'.format(key=key) for key in data])
        sql += update
        try:
            if self.cursor.execute(sql, tuple(data.values())*2):
                print("successful")
                self.db.commit()
        except Exception as e:
            print(e)
            self.db.rollback()

    def delete(self):
        condition = "age > 20"
        sql = "delete from {table} where {condition}".format(table=self.table, condition=condition)
        try:
            self.cursor.execute(sql)
            self.db.commit()
        except Exception as e:
            print(e)
            self.db.rollback()

    def query(self):
        sql = "select * from {table} where age < 20".format(table=self.table)
        try:
            self.cursor.execute(sql)
            print("count: ", self.cursor.rowcount)
            one = self.cursor.fetchone()
            print("one: ", one)
            results = self.cursor.fetchall()
            print("results: ", results)
            print("results type: ", type(results))
            for row in results:
                print(row)
        except Exception as e:
            print(e)

    def close(self):
        self.db.close()

if __name__ == '__main__':
    mysql = MySQLUtil()
    mysql.connect_db()
    mysql.create_table("students")
    mysql.insert("20180605", "futhead", 28)
    mysql.common_insert()
    mysql.update()
    mysql.upsert()
    mysql.delete()
    mysql.query()
    mysql.close()