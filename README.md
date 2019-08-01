# predict_mlb

## Let's predict the major league baseball team's odds for the playoff.

MLB.com provides the stats of the AL/NL teams such as Batting average, OPS (On-base  Plus Slugging), ERA (Earned Runs Average), WHIP(Walks Plus Hits Divided by Innings Pitched), etc.

And there are API that provides get some stats of MLB with Python. You need to install it with 
``` pip install MLB-StatsAPI ```

Based on the Hitting, Pitching, and Fielding stats for division winners of the last 10 years,  let's try to make a machine learning model with classification methods. (Logistic Regression)

With this model, let's figure out which team is going to be advanced to the playoff.
