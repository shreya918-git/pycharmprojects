from flask import Flask,render_template,request
from users import User
from intent import query
app=Flask(__name__)
@app.route("/")
def index():
    return render_template("login.html")
@app.route("/signup")
def signup():
    return render_template("signup.html")
@app.route("/perform_signup",methods=["POST"])
def perform_signup():
 name=request.form.get("name")
 email=request.form.get("email")
 password=request.form.get("password")
 user=User()
 response=user.signup(name,email,password)
 if response:
    return render_template("login.html")
 else:
    return "signup failed"
@app.route("/perform_login",methods=["POST"])
def perform_login():
 email=request.form.get("email")
 password=request.form.get("password")
 user=User()
 response=user.login(email,password)
 if response:
    return render_template("ner.html")
 else:
    return "login failed"


@app.route("/perform_intent", methods=["POST"])
def perform_intent():
    text = request.form.get("text")
    payload = {
        "inputs": text,
        "parameters": {"candidate_labels": ["refund", "complaint", "faq", "feedback"]}
    }
    response = query(payload)

    if response:
        # Zip labels and scores together here
        intent_results = zip(response["labels"], response["scores"])
        return render_template("ner.html", intent_results=intent_results)
    else:
        return render_template("ner.html", intent_results=None)




if __name__ == "__main__":
    app.run(debug=True)