class Config(object):
    SECRET_KEY = "abcdefg"
    SQLALCHEMY_DATABASE_URI = "mysql+pymysql://root:futhead@localhost:3306/flaskblog?charset=utf8"
    SQLALCHEMY_TRACK_MODIFICATIONS = False