import json
class User:
    def login(self,a,b):
        with open("db.json","r") as rf:
            database=json.load(rf)
        if a in database:
            if b in database[a]:
                return 1
            else:
                return 0
        else:
            return 0
    def signup(self,a,b,c):
        with open("db.json","r") as rf:
            database=json.load(rf)
        if b in database:
            return 0
        else:
            database[b]=[a,c]
            with open("db.json","w") as wf:
                json.dump(database,wf)
                return 1