import os
os.environ['DATABASE_URL'] = 'sqlite:///./test.db'

from app.database import SessionLocal
from app.models import User

db = SessionLocal()
user = db.query(User).filter(User.email == "oswaldoadw18@gmail.com").first()
if user:
    user.plan = "PRO"
    db.commit()
    print(f"Usuário {user.email} atualizado para plano PRO")
    print(f"Agora você tem limite de 10 condomínios!")
else:
    print("Usuário não encontrado")
db.close()
