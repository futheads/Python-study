from app import db
from app.models import User, Post

# u = User(username="fuzhouzhou", email="fuzhouzhou@163.com")
# db.session.add(u)
# db.session.commit()

# users = User.query.all()
# for u in users:
#     print(u)

# print(User.query.get(2))

# u = User.query.get(1)
# p = Post(body="我第二次提交的数据！", author=u)
# db.session.add(p)
# db.session.commit()

# posts = u.posts.all()
# print(User.query.order_by(User.username.desc()).all())

# db.session.delete(User.query.get(2))
# for user in User.query.all():
#     for post in user.posts:
#         db.session.delete(post)
#     db.session.delete(user)
#     db.session.commit()


u = User(username="futhead", email="futhead@qq.com")
u.set_password("futhead")
u.check_password("fuzhouzhou")
print(u.password_hash)