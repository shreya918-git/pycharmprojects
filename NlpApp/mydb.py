import json
class Database:
    def add_value(self,a,b,c):
        with open("resources/db.json","r") as rf:
            database=json.load(rf)
        if b in database:
            return 0
        else:
            database[b]=[a,c]
            with open("resources/db.json","w") as wf:
                json.dump(database,wf)
            return 1
    def login(self,a,b):
        with open("resources/db.json","r") as rf:
            database=json.load(rf)
        if a in database:
            if database[a][1]==b:
                return 1
            else:
                return 0
        else:
            return 0