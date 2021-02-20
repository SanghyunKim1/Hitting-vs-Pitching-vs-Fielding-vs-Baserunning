# Hitting vs Pitching vs Fielding vs Baserunning
## Content
1. Intro
2. Metadata
3. Data Cleaning
4. EDA (Exploratory Data Analysis)
5. Feature Scaling
6. Random Forest Regressoin
7. Feature Importance
8. Cross-era Comparison
9. Multiple Linear Regression
10. Conclusion

### 1. Intro
Baseball is a complicated sport that consists of many different factors such as player skills, team chemistry, health, money, weather and so on (*even luck as well*). And these various factors can be broken down as follows: what we can measure and predict, and what we cannot. For example, we can easily measure skills and team payrolls, while it's relatively hard (may be impossible) to accurately measure weather and luck.

Then one of the most reasonable questions raised by baseball operators and fans would be **How much of each factor contributes a team's winning percentage?**
In this analysis, I focused on what I can predict: **players' pure skills**. The reason I excluded other factors is simple. *They're either unavailable or unmeasurable*. Team payroll data for old teams (1870s ~ 1950s) are unavailable, and team chemistry, impacts of weather on baseball or luck are unmeasaurable.

Players' pure skills can further be broke down into four different aspects: **Hitting**, **Pitching**, **Fielding**, and **Baserunning**. To find the percentage contributions of these factors to a team's winning percentage, I ran a random forest regression model and used an impurity-based feature importance method.

### 2. Metadata
| **Metadata** | **Information** |
| :-----------: | :-----------: |
| **Origin of Data** | [FanGraphs.com](https://www.fangraphs.com) |
| **Terms of Use** | [Terms and Conditions](https://www.fangraphs.com/about/terms-of-service) |
| **Data Structure** | a team batting dataset consisting of 2926 rows * 20 columns |
| **Data Structure** | a team pitching dataset consisting of 2926 rows * 21 columns |

| **Batting Data Feature** | **Data Meaning** |
| :-----------: | :-----------: |
| ***Season*** | Each year refers to corresponding seasons from 1871 to 2019 |
| ***Team*** | All the Major League Baseball Teams |
| ***PA*** | [Plate Appearance](http://m.mlb.com/glossary/standard-stats/plate-appearance) |
| ***HR*** | [Home run](http://m.mlb.com/glossary/standard-stats/home-run) |
| ***R*** | [Runs Scored](http://m.mlb.com/glossary/standard-stats/run) |
| ***RBI*** | [Runs Batted In](http://m.mlb.com/glossary/standard-stats/runs-batted-in) |
| ***SB*** | [Stolen Base](http://m.mlb.com/glossary/standard-stats/stolen-base) |
| ***ISO*** | [Isolated Power](https://library.fangraphs.com/offense/iso/) |
| ***BABIP*** | [Batting Average on Balls in Play](https://library.fangraphs.com/pitching/babip/) |
| ***AVG*** | [Batting Average](http://m.mlb.com/glossary/standard-stats/batting-average) |
| ***OBP*** | [On-base Percentage](https://library.fangraphs.com/offense/obp/) |
| ***SLG*** | [Slugging Percentage](http://m.mlb.com/glossary/standard-stats/slugging-percentage) |
| ***wOBA*** | [Weighted On-Base Average](https://library.fangraphs.com/offense/woba/) |
| ***wRAA*** | [Weighted Runs Above Average](https://library.fangraphs.com/offense/wraa/) |
| ***wRC+*** | [Weighted Runs Created Plus](https://library.fangraphs.com/offense/wrc/) |
| ***EV*** | [Exit Velocity](http://m.mlb.com/glossary/statcast/exit-velocity) |
| ***BsR*** | [Base Running Runs](https://library.fangraphs.com/offense/bsr/) |
| ***Off*** | [Offensive Runs Above Average](https://library.fangraphs.com/offense/off/) |
| ***Def*** | [Defensive Runs Above Average](https://library.fangraphs.com/defense/def/) |
| ***WAR*** | [Wins Above Replacement for Batters](https://library.fangraphs.com/war/war-position-players/) |

| **Pitching Data Feature** | **Data Meaning** |
| :-----------: | :-----------: |
| ***Season*** | Each year refers to corresponding seasons from 1871 to 2019 |
| ***Team*** | All the Major League Baseball Teams |
| ***W*** | Number of Wins |
| ***L*** | Number of Losses |
| ***SV*** | Number of Saves |
| ***IP*** | [Innings Pitched](http://m.mlb.com/glossary/standard-stats/innings-pitched) |
| ***R*** | [Runs Allowed](http://m.mlb.com/glossary/standard-stats/run) |
| ***ER*** | [Earned Runs](http://m.mlb.com/glossary/standard-stats/earned-run) |
| ***H/9*** | [Hits per 9 Innings](http://m.mlb.com/glossary/advanced-stats/hits-per-nine-innings) |
| ***K/9*** | [Strikeouts per 9 Innings](https://library.fangraphs.com/pitching/rate-stats/) |
| ***BB/9*** | [Walks per 9 Innings](https://library.fangraphs.com/pitching/rate-stats/) |
| ***HR/9*** | [Homeruns per 9 Innings](http://m.mlb.com/glossary/advanced-stats/home-runs-per-nine-innings) |
| ***WHIP*** | [Walks plus Hits per Innings Pitched](https://library.fangraphs.com/pitching/whip/) |
| ***BABIP*** | [Batting Average on Balls in Play](https://library.fangraphs.com/pitching/babip/) |
| ***EV*** | [Exit Velocity](http://m.mlb.com/glossary/statcast/exit-velocity) |
| ***ERA*** | [Earned Runs Average](https://library.fangraphs.com/pitching/era/) |
| ***FIP*** | [Fielding Independent Pitching](https://library.fangraphs.com/pitching/fip/) |
| ***xFIP*** | [Expected Fielding Independent Pitching](https://library.fangraphs.com/pitching/xfip/) |
| ***RAR*** | Runs Above Replacement for Pitchers |
| ***RA9-WAR*** | [Runs Allowed based WAR](https://library.fangraphs.com/pitching/fdp/) |
| ***WAR*** | [Wins Above Replacement for Pitchers](https://library.fangraphs.com/war/calculating-war-pitchers/) |
