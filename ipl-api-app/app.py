from flask import Flask,jsonify,request
import ipl
import functions
app=Flask(__name__)
@app.route("/")
def index():
    return "Hello World"
@app.route("/api/teams")
def teams():
    response=ipl.total_teams()
    return jsonify(response)
@app.route("/api/TeamVsTeam")
def TeamVsTeam():
    team1=request.args.get("team1")
    team2=request.args.get("team2")
    if team1 in ipl.total_teams() and team2 in ipl.total_teams():
       response=ipl.teamvsteam(team1,team2)
       return jsonify(response)
    else:
        return {"message":"invalid team name"}
@app.route("/api/TeamRecord")
def TeamRecord():
    team=request.args.get("team")
    response=functions.teamAPI(team)
    return response
@app.route("/api/BatterRecord")
def BatterRecord():
    batter=request.args.get("batter")
    response=functions.batterAPI(batter)
    return response
@app.route("/api/BowlerRecord")
def BowlerRecord():
    bowler=request.args.get("bowler")
    response=functions.bowlerAPI(bowler)
    return response
if __name__=="__main__":
    app.run(debug=True)