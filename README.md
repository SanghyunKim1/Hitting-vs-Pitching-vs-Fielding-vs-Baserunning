# Hitting vs Pitching vs Fielding vs Baserunning
## Content
1. Intro
2. Metadata
3. Data Cleaning
4. Feature Selection (Domain Knowledge)
5. EDA (Exploratory Data Analysis)
6. Random Forest Regressoin
7. Feature Importance
8. Cross-era Comparison
9. Multiple Linear Regression
10. Conclusion

### 1. Intro
Baseball is a complicated sport that consists of many different factors such as player skills, team chemistry, health, money, weather and so on (*even luck as well*). And these various factors can be broken down as follows: what we can measure and predict, and what we cannot. For example, we can easily measure skills and team payrolls, while it's relatively hard (may be impossible) to accurately measure weather and luck.

Then one of the most reasonable questions raised by baseball operators and fans would be **How much of each factor contributes to a team's winning percentage?**
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

### 3. Data Cleaning
- Renamed data features for clarity.
  * *Batting Dataset*: **R** to **RS**
  * *Batting Dataset*: **WAR** to **bWAR**
  * *Pitching Dataset*: **R** to **RA**
  * *Pitching Dataset*: **BABIP** to **oppBABIP**
  * *Pitching Dataset*: **EV** to **oppEV**
  * *Pitching Dataset*: **WAR** to **pWAR**
- Combined team batting and team pitching dataset.
- Created new data features.
  * **UER**: *Unearned Runs* = **RA** - **ER**
  * **G**: *# Games Played* = **W** + **L**
  * **wPCT**: *Team Winning Percentage* = (**W** + **L**)/**G**
- Checked missing data and replaced them **xFIP** projected values based on linear regression result (**IterativeImputer**).
- Dropped unncessary columns.

### 4. Feature Selection (Domain Knowledge)
In this analysis, I selected features (*independent variables* to predict team winning percentages) based on domain knowledge.
As I briefly mentioned above, the purpose of this analysis is to quantify the percentage contribution of player skills (i.e. **Hitting**, **Pitching**, **Fielding** and **Baserunning**). To find statistics that accurately measure players' skills, I focused on *(i) what's the primary job of players in terms of those 4 different aspects*, *(ii)how each stat is calculated*, and *(iii) what these stats measure in the end*. The logic for selected features is the following:

- **Hitting**: For hitting, the primary job of hitters is to score runs (or drive runs) by reaching bases and advancing runners on bases. Therefore, I needed a stat that only quantifies the **pure hitting ability** but excludes all other things such as a baserunning ability, or fielding ability. In this regard, the perfect (at least the best for now) fit would be **wOBA**. Unlike other conventional stats like *SLG* or *OPS*, **wOBA** assigns different weights to each hitting event (*single*, *double*, *tripple*, *home run*, *walks* etc.) in terms of run production.
One of the most common problem is that conventional stats have is they treat different hitting events equally (e.g. *AVG*) or they give wrong weights to each hitting event (e.g. *SLG*). Since it assigns proper weights to each hitting event based on *Run Production*, it also reflects the changes in run production in each season (i.e. the weights assigned to hitting events differe from season to season). Thus, at the time of writing, wOBA is considered the most powerful hitting stat. For more information about wOBA, see [here](https://library.fangraphs.com/offense/woba/).

- **Pitching**: Unlike hitting, finding a single stat that properly represents the pitching ability is way more difficult and troublesome. There's still ongoing discussions about seperating pitching ability and the fielding ability. We should keep in mind that when pitchers are on the mound, fielders also do their jobs not to allow runs, and thus, it makes diffcult to properly quantify the contribution of these two jobs. One attempt to separate pitching and fielding is [*DIPS (Defene Independent Pitching Statistics)*](https://library.fangraphs.com/principles/dips/) devised by *Voros McCracken*. The logic is simple. According to *DIPS*, pitchers' have no control of the balls put into play, and therefore, pitchers should be judged by only what they can control: **(i) Strikeouts**, **(ii) Walks**, and **(iii) Home Runs**.
**FIP** is one of the most popular *DIPS-based* stat that gives different weights to those three pitching events. Although I somewhat disagree with the idea that pitchers have no responsibility for balls put into play at all, other conventional stats like *ERA* or *WHIP* fail to separate pitching and fielding, so I used FIP to measure a team's **pure pitching ability**. There's another stat called *SIERA* which seems to resolve limitations that FIP has but this stat is only available from the 2002 season, I decided not to use it. For more information about FIP, see [here](https://library.fangraphs.com/pitching/fip/).

- **Fielding**: Fielding is a relatively more recent research area where sabermetricians still try to find better stats. The criteria to find a single fielding measurement is: (i)it should be an overall stat that properly represents a team's fielding ability, and (ii) it should be avaiable for old teams like 1870s teams. Based on these two criteria, I used **Def** devised by *FanGraphs*.
**Def** is the sum of *fielding runs above average* and *positional adjustment*. The *fielding runs above avarage* here is measured by *UZR (Ultimate Zone Rating)* which measures a fielder's run values compared to the probability that an average MLB fielder would successfully convert balls into outs at nine different positions. For more information about Def, see [here](https://library.fangraphs.com/defense/def/).

- **Baserunning**: Unlike other aspects of player skills, baserunning is an area where there are not many stats available. Some sabermetricians measure a runner's sprint speed (in ft/sec), but such data are unavailable for the past day's teams and inappropriate to use for a team's base running ability. Therefore, I used **BsR** devised by *FanGraphs*. According to *FanGraphs*, **BsR** is an overall base running stat that turns *(i) weighted stolen bases*, *(ii) weighted grounded into double play runs*, and *(iii) Ultimate Base Running (UBR)* into runs above/below average. For more information about BsR, see [here](https://library.fangraphs.com/offense/bsr/).
 

### 5. EDA (Exploratory Data Analysis)
***5-1. Normality***
![](https://github.com/shk204105/Hitting-vs-Pitching-vs-Fielding-vs-Baserunning/blob/master/images/wPCT%20Histogram:Q-Q%20Plot.png)

<table>
<tr>
<td><img src="https://github.com/shk204105/Hitting-vs-Pitching-vs-Fielding-vs-Baserunning/blob/master/images/Histograms.png" width="600" height="450"></td>
<td><img src="https://github.com/shk204105/Hitting-vs-Pitching-vs-Fielding-vs-Baserunning/blob/master/images/Q-Q%20Plots.png" width="600" height="450"></td>
</tr>
</table> 

According to the histograms and Q-Q plots above, although all the features seem to follow approximate noraml distributions, **wOBA** and **BsR** are slighty skewed. However, as I'm going to use a random forest regression model, normalizing data is unnecessary.


***5-2. Scaling***

<img src="https://github.com/shk204105/Hitting-vs-Pitching-vs-Fielding-vs-Baserunning/blob/master/images/KDE%20Plot.png" width=600 height=600>

Although scaling is also not needed for random forest models, I scaled features using *StandardScaler* to get some ideas about feature importance by directly comparing features.

***5-3. Historical Changes in Each Stat***

Since 1871, many external factors (e.g. changes in rules and resilience of the ball, league expansion, or advances in skills) have been affecting the way games are played (i.e. how teams win the ball game). Therefore, it's reasonable to think that such external factors must have affected league average stats throughout the MLB history. Further, looking at those historical changes in each stat would also give us some general ideas about what was the most important factors in different eras. To see those changes I created two time series plots.

*Note: I used median values of each stat instead of mean values to avoid outlier issues. Also, I used scaled data to accurately measure 

![](https://github.com/shk204105/Hitting-vs-Pitching-vs-Fielding-vs-Baserunning/blob/master/images/Time%20Series%20Plot1.png)

According to the line plot above, while *Def* and *BsR* seem to stay constant, *wOBA* and *FIP* that represent **Hitting** and **Pitching**, respectively are relatively more fluctuating depending on eras. What does it indicate?. Well, it could show that *the importance of **hitting** and **pitching** ability has been larger than **fielding** and **baserunning***.

Another pattern we can see from this plot is that when league wOBA was relatively high (i.e. hitter-friendly eras), league FIP also got higher (i.e. pitchers must have struggled with doing their jobs during the same eras) on average, and *vice versa*. (Note: the lower FIP is the better pitchers do their jobs.)

![](https://github.com/shk204105/Hitting-vs-Pitching-vs-Fielding-vs-Baserunning/blob/master/images/Time%20Series%20Plot2.png)

I also created year bins to see a general trend for those stats, and the result seems similar to what I've concluded above.

***5-4. Winning Team vs Losing Team***

![](https://github.com/shk204105/Hitting-vs-Pitching-vs-Fielding-vs-Baserunning/blob/master/images/Bar%20Plot.png)

Another way to get some ideas about feature importance would be comparing how those four stats differ on average based on team winning percentages. To do so, I created a binary feature which represents whether a team winning percentage is above 0.500 or below 0.500, and compared how the average values of stats differ depending on these two groups.

The bar plot above where each bar represents the median scales of stats depicts that there's a notable difference in stats between these two groups. Teams of which winning percentage is above 0.500 records higher team *wOBA*, *Def* and *BsR* with lower *FIP* than teams with winning percentage below 0.500. However, it's not the only thing this bar plot shows.

See how large the differences in each stat between these two groups are. While the differences in *wOBA* and *Def* are relatively larger, the difference in *FIP* is not as significant as *wOBA* and *Def*. Moreoever, the difference in *BsR* between these two groups are marginal compared to other three stats. Thus, **hitting** and **fielding** might have more significant impacts on a team's winning percentage than **pitching**, while **baserunning** is not that important. Let's see if that's the case.

### 6. Random Forest Regression

![](https://github.com/shk204105/Hitting-vs-Pitching-vs-Fielding-vs-Baserunning/blob/master/images/Estimators.png)

