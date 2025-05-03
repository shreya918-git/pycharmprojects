from flask import Flask,render_template,request
import requests
app=Flask(__name__)
@app.route("/")
def index():
    team=requests.get("http://127.0.0.1:5000/api/teams")
    return render_template("index.html",teams=team.json())
@app.route("/perform_analysis",methods=["GET"])
def perform_analysis():
    team1=request.args.get("team1")
    team2=request.args.get("team2")
    try:
        response = requests.get("http://127.0.0.1:5000/api/TeamVsTeam", params={"team1": team1, "team2": team2})
        response.raise_for_status()  # Raises HTTPError for bad status codes
        data = response.json()  # May raise JSONDecodeError
    except requests.exceptions.RequestException as e:
        print("Request failed:", e)
        data = {"error": "API request failed"}
    except ValueError:
        print("Invalid JSON in response")
        data = {"error": "Invalid JSON format"}
    team = requests.get("http://127.0.0.1:5000/api/teams")
    return render_template("index.html",response=data,teams=team.json())
app.run(debug=True,port=7000)