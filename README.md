# Hitting vs Pitching vs Fielding vs Baserunning
## Table of Contents
1. Intro
2. Technologies
3. Metadata
4. Data Cleaning
5. Feature Selection (Domain Knowledge)
6. EDA (Exploratory Data Analysis)
7. Random Forest Regressoin with Feature Importance
8. Cross-era Comparison
9. Conclusion

### 1. Intro
Baseball is a complicated sport that consists of many different factors such as player skills, team chemistry, health, money, weather and so on (*even luck as well*). And these various factors can be broken down as follows: what we can measure and predict, and what we cannot. For example, we can easily measure skills and team payrolls, while it's relatively hard (may be impossible) to accurately measure weather and luck.

Then one of the most reasonable questions raised by baseball operators and fans would be **How much of each factor contributes to a team's winning percentage?**
In this analysis, I focused on what I can predict: **players' pure skills**. The reason I excluded other factors is simple. *They're either unavailable or unmeasurable*. Team payroll data for old teams (1870s ~ 1950s) are unavailable, and team chemistry, impacts of weather on baseball or luck are unmeasaurable.

Players' pure skills can further be broke down into four different aspects: **Hitting**, **Pitching**, **Fielding**, and **Baserunning**. To find the percentage contributions of these factors to a team's winning percentage, I ran a random forest regression model and used an impurity-based feature importance method.

### 2. Technologies
- Python 3.8
  * Pandas - version 1.2.2
  * Numpy - version 1.20.1
  * matplotlib - version 3.3.4
  * seaborn - version 0.11.1
  * scikit-learn - version 0.24.1
  * statsmodels - version 0.12.2
  * scipy - version 1.6.1
  * missingno - version 0.4.2

### 3. Metadata
| **Metadata** | **Information** |
| :-----------: | :-----------: |
| **Origin of Data** | [FanGraphs](https://www.fangraphs.com) |
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
| ***HR/9*** | [Home runs per 9 Innings](http://m.mlb.com/glossary/advanced-stats/home-runs-per-nine-innings) |
| ***WHIP*** | [Walks plus Hits per Innings Pitched](https://library.fangraphs.com/pitching/whip/) |
| ***BABIP*** | [Batting Average on Balls in Play](https://library.fangraphs.com/pitching/babip/) |
| ***EV*** | [Exit Velocity](http://m.mlb.com/glossary/statcast/exit-velocity) |
| ***ERA*** | [Earned Runs Average](https://library.fangraphs.com/pitching/era/) |
| ***FIP*** | [Fielding Independent Pitching](https://library.fangraphs.com/pitching/fip/) |
| ***xFIP*** | [Expected Fielding Independent Pitching](https://library.fangraphs.com/pitching/xfip/) |
| ***RAR*** | Runs Above Replacement for Pitchers |
| ***RA9-WAR*** | [Runs Allowed based WAR](https://library.fangraphs.com/pitching/fdp/) |
| ***WAR*** | [Wins Above Replacement for Pitchers](https://library.fangraphs.com/war/calculating-war-pitchers/) |

### 4. Data Cleaning
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
  * **G**: *Number of Games Played* = **W** + **L**
  * **wPCT**: *Team Winning Percentage* = (**W** + **L**)/**G**
- Checked missing data and replaced **xFIP** projected values based on linear regression result (**IterativeImputer**).
- Dropped unncessary columns.

### 5. Feature Selection (Domain Knowledge)
In this analysis, I selected features (*independent variables* to predict team winning percentages) based on domain knowledge.
As I briefly mentioned above, the purpose of this analysis is to quantify the percentage contribution of player skills (i.e. **Hitting**, **Pitching**, **Fielding** and **Baserunning**). To find statistics that accurately measure players' skills, I focused on *(i) what's the primary job of players in terms of those 4 different aspects*, *(ii)how each stat is calculated*, and *(iii) what these stats measure in the end*. The logic for selected features is the following:

- **Hitting**: For hitting, the primary job of hitters is to score runs (or drive runs) by reaching bases and advancing runners on bases. Therefore, I needed a stat that only quantifies the **pure hitting ability** but excludes all other things such as a baserunning ability or fielding ability. In this regard, the perfect (at least the best for now) fit would be **wOBA**. Unlike other conventional stats like *SLG* or *OPS*, **wOBA** assigns different weights to each hitting event (*single*, *double*, *triple*, *home run*, *walks* etc.) in terms of run production.
One of the most common problem is that conventional stats have is they treat different hitting events equally (e.g. *AVG*) or they give wrong weights to each hitting event (e.g. *SLG*). Since it assigns proper weights to each hitting event based on *Run Production*, it also reflects the changes in run production in each season (i.e. the weights assigned to hitting events differ from season to season). Thus, at the time of writing, wOBA is considered the most powerful hitting stat. For more information about wOBA, see [here](https://library.fangraphs.com/offense/woba/).

- **Pitching**: Unlike hitting, finding a single stat that properly represents the pitching ability is way more difficult and troublesome. There's still ongoing discussions about separating pitching ability and the fielding ability. We should keep in mind that when pitchers are on the mound, fielders also do their jobs not to allow runs, and thus, it makes difficult to properly quantify the contribution of these two jobs. One attempt to separate pitching and fielding is [*DIPS (Defene Independent Pitching Statistics)*](https://library.fangraphs.com/principles/dips/) devised by *Voros McCracken*. The logic is simple. According to *DIPS*, pitchers' have no control of the balls put into play, and therefore, pitchers should be judged by only what they can control: **(i) Strikeouts**, **(ii) Walks**, and **(iii) Home Runs**.
**FIP** is one of the most popular *DIPS-based* stat that gives different weights to those three pitching events. Although I somewhat disagree with the idea that pitchers have no responsibility for balls put into play at all, other conventional stats like *ERA* or *WHIP* fail to separate pitching and fielding, so I used FIP to measure a team's **pure pitching ability**. There's another stat called *SIERA* which seems to resolve limitations that FIP has but this stat is only available from the 2002 season, I decided not to use it. For more information about FIP, see [here](https://library.fangraphs.com/pitching/fip/).

- **Fielding**: Fielding is a relatively more recent research area where sabermetricians still try to find better stats. The criteria to find a single fielding measurement is: (i)it should be an overall stat that properly represents a team's fielding ability, and (ii) it should be available for old teams like 1870s teams. Based on these two criteria, I used **Def** devised by *FanGraphs*.
**Def** is the sum of *fielding runs above average* and *positional adjustment*. The *fielding runs above average* here is measured by *UZR (Ultimate Zone Rating)* which measures a fielder's run values compared to the probability that an average MLB fielder would successfully convert balls into outs at nine different positions. For more information about Def, see [here](https://library.fangraphs.com/defense/def/).

- **Baserunning**: Unlike other aspects of player skills, baserunning is an area where there are not many stats available. Some sabermetricians measure a runner's sprint speed (in ft/sec), but such data are unavailable for the past day's teams and inappropriate to use for a team's base running ability. Therefore, I used **BsR** devised by *FanGraphs*. According to *FanGraphs*, **BsR** is an overall base running stat that turns *(i) weighted stolen bases*, *(ii) weighted grounded into double play runs*, and *(iii) Ultimate Base Running (UBR)* into runs above/below average. For more information about BsR, see [here](https://library.fangraphs.com/offense/bsr/).
 

### 6. EDA (Exploratory Data Analysis)
***6-1. Normality***
![](https://github.com/shk204105/Hitting-vs-Pitching-vs-Fielding-vs-Baserunning/blob/master/images/wPCT%20Histogram:Q-Q%20Plot.png)

<table>
<tr>
<td><img src="https://github.com/shk204105/Hitting-vs-Pitching-vs-Fielding-vs-Baserunning/blob/master/images/Histograms.png" width="600" height="450"></td>
<td><img src="https://github.com/shk204105/Hitting-vs-Pitching-vs-Fielding-vs-Baserunning/blob/master/images/Q-Q%20Plots.png" width="600" height="450"></td>
</tr>
</table> 

According to the histograms and Q-Q plots above, although all the features seem to follow approximate noraml distributions, **wOBA** and **BsR** are slighty skewed. However, as I'm going to use a random forest regression model, normalizing data is unnecessary.


***6-2. Scaling***

<img src="https://github.com/shk204105/Hitting-vs-Pitching-vs-Fielding-vs-Baserunning/blob/master/images/KDE%20Plot.png" width="600" height="600">

Although scaling is also not needed for random forest models, I scaled features using *StandardScaler* to get some ideas about feature importance by directly comparing features.

***6-3. Historical Changes in Each Stat***

Since 1871, many external factors (e.g. changes in rules and resilience of the ball, league expansion, or advances in skills) have been affecting the way games are played (i.e. how teams win the ball game). Therefore, it's reasonable to think that such external factors must have affected league average stats throughout the MLB history. Further, looking at those historical changes in each stat would also give us some general ideas about what was the most important factors in different eras. To see those changes I created two time series plots.

*Note: I used median values of each stat instead of mean values to avoid outlier issues. Also, I used scaled data to directly compare different stats.

![](https://github.com/shk204105/Hitting-vs-Pitching-vs-Fielding-vs-Baserunning/blob/master/images/Time%20Series%20Plot1.png)

According to the line plot above, while *Def* and *BsR* have been staying constant, *wOBA* and *FIP* that represent **Hitting** and **Pitching**, respectively are relatively more fluctuating depending on eras. What does it indicate? Well, it could show that *the importance of **hitting** and **pitching** ability has been larger than **fielding** and **baserunning***.

Another pattern we can see from this plot is that when league wOBA was relatively high (i.e. hitter-friendly eras), league FIP also got higher (i.e. pitchers must have struggled with doing their jobs during the same eras) on average, and *vice versa*. (Note: the lower FIP is the better pitchers do their jobs.)

![](https://github.com/shk204105/Hitting-vs-Pitching-vs-Fielding-vs-Baserunning/blob/master/images/Time%20Series%20Plot2.png)

I also created year bins to see a general trend for those stats, and the result seems similar to what I've concluded above.

***6-4. Winning Team vs Losing Team***

![](https://github.com/shk204105/Hitting-vs-Pitching-vs-Fielding-vs-Baserunning/blob/master/images/Bar%20Plot.png)

Another way to get some ideas about feature importance would be comparing how those four stats differ on average based on team winning percentages. To do so, I created a binary feature which represents whether a team winning percentage is above 0.500 or below 0.500, and compared how the average values of stats differ depending on these two groups.

The bar plot above where each bar represents the median scales of stats depicts that there's a notable difference in stats between these two groups. Teams of which winning percentage is above 0.500 records higher team *wOBA*, *Def* and *BsR* with lower *FIP* than teams with winning percentage below 0.500. However, it's not the only thing this bar plot shows.

See how large the differences in each stat between these two groups are. While the differences in *wOBA* and *Def* are relatively larger, the difference in *FIP* is not as significant as *wOBA* and *Def*. Moreover, the difference in *BsR* between these two groups are marginal compared to other three stats. Thus, **hitting** and **fielding** might have more significant impacts on a team's winning percentage than **pitching**, while **baserunning** is not that important. Let's see if that's the case.

### 7. Random Forest Regression with Feature Importance

<img src="https://github.com/shk204105/Hitting-vs-Pitching-vs-Fielding-vs-Baserunning/blob/master/images/Random%20Forest.png" width="650" height="400">

***7-1. Random Forest Regression***

The random forest algorithm is an ensemble learning technique that combines predictions from multiple decision trees to make more accurate predictions than an individual decision tree algorithm does. As random forests use a bootstrap aggregation (bagging) method, it runs individual decision trees and aggregates the outputs without any biased preference to any model at the end. For this reason, it resolves the weakness of decision trees, and therefore, it's one of the most frequently used machine learning techniques for both classification and regression tasks.

![](https://github.com/shk204105/Hitting-vs-Pitching-vs-Fielding-vs-Baserunning/blob/master/images/Estimators.png)

To run a random forest model, I need to decide the number of indibidual trees that can yield the best result. Using the *for loop*, the best number of estimators that yield the highest score in the random forest regression model is turned out to be *190*. With 190 individual decision trees, the random forest regression result is:

| **Measurement** | **Score** | 
| :-----------: | :-----------: |
| ***R-squared*** | 0.550845576425548 |
| ***RMSE*** | 0.06439063966080207 |

Though the R-squared of 0.55 is not that impressive, the R-squared in this project isn't as important as it is for prediction tasks  because the goal of this project is to find relationships between variables. The RMSE of 0.064 conveys that the average difference between predictive winnig percentages and observed winning percentages is about 0.064. A winning percentage of 0.064 is equivalent to 10.3 games in modern baseball (162 games * 0.064). Thus, the difference of ± 5 games would give us reasonable predictions given the number of total games for one season (162 games in total).

***7-2. Random Forest Feature Importance***

| **Feature** | **Feature Importance** | 
| :-----------: | :-----------: |
| ***wOBA*** | 0.335 |
| ***FIP*** | 0.298 |
| ***Def*** | 0.255 |
| ***BsR*** | 0.111 |

Throughout the MLB history (1871 ~ 2019), it appears that **hitting** has the most significant impact on a team's winning percentage (about 33.5%), **pitching**  (about 29.8%), **fielding** (about 25.5%) have the second most impacts, and **baserunning** (about 11.1%) is the least important. The result seems somewhat similar to what I've concluded in *EDA*. Nevertheless, as I mentioned above, there have been various external factors that affected the ball game throughout the MLB history. Therefore, I also ran random forest regression models to see how feature importance varied from era to era.

### 8. Cross-era comparison

With the same data, the result of random forest models in each era is:

| **Measurement** | **1871-1899** | **1900-1919** | **1920-1939** | **1940-1959** | **1960-1979** | **1980-1999** | **2000-2019** |  
| :-----------: | :-----------: | :-----------: | :-----------: | :-----------: | :-----------: | :-----------: | :-----------: | 
| ***R-squared*** | 0.573 | 0.780 | 0.710 | 0.816 | 0.743 | 0.673 | 0.714 |
| ***RMSE*** | 0.099 | 0.047 | 0.051 | 0.040 | 0.038 | 0.039 | 0.040 |

According to the result, the random forest models seem to yield a better result in each era with higher R-squared values got higher and lower RMSEs (except the early days of MLB).

![](https://github.com/shk204105/Hitting-vs-Pitching-vs-Fielding-vs-Baserunning/blob/master/images/Feature%20Importance.png)

To effectively see how feature importance varied from era to era, I combined all the results in each era and visualized discrete distributions in a horizontal bar chart. According to the bar chart above, there is a clear trend in feature importance from era to era.

In the early days (1871 ~ 1899), **fielding** had the most significant impacts on a team's winnig percentage (about 60.2%), while **hitting**, **pitching** and **baserunning** had marginal influences. However, it appears that such a result has been reversed as time goes on. The importance of **fielding** has declined throughout the history, whereas the importance of **hitting** and **pitching** has increased recently. In the 2000s, **hitting** and **pitching** have the majority part of a team's winning percentage (about 80% together), while **fielding** and **baserunning** don't.

Furthermore, there's one more clear pattern here. **Baserunning** has never been as important as hitting, pitching, and fielding since the start of MLB. In other words, teams have never benefitted that much from successful base running since 1871, so why do they want to focus on base running?

### 9. Conclusion

In this project, I analyzed how important each aspect of player skills is in terms of a team's winning percentage. The four main areas are **hitting**, **pitching**, **fielding** and **baserunning**, and these skills are measured by *wOBA*, *FIP*, *Def* and *BsR*, respectively. Of course, single stat wouldn't be enough to perfectly evaluate each areas of player abilities. Nevertheless, as this project aims to understand the impacts of *pure* skills, I decided to include single stats for each aspect to avoid overestimation problems, and therefore, the models above could yeild relatively small R-squared values. Though it doesn't mean that I totally disregarded the predictive power of models. Given the small RMSEs of models in all eras, I believe those models still yield a decent level of predictions (see the RMSE table above).

Through EDA and random forest feature importance (*impurity-based method*) analysis, I came to a conclusion that **fielding** used to be the most significant factor for teams in the early days, but such a trend seems to have changed over time. Since the 1900s, **hitting** and **pitching** started to account for large proportions of team winning percentages, while **fielding** is getting less important than it used to be. On the other hand, **baserunning** has never influenced a team's winning percentage that much. So it appears that teams would benefit from strengthening their hitting and pitching capacities.

Nonetheless, this analysis also has several drawbacks. First, the impurity-based method for measuring feature importance tend to overestimate numerical features. Thus, the percentage contributions of each stat might have been overestimated. Second, since I didn't use post-season data, the result may be inappropriate to conclude how much importance those stats have for post-season results. Therefore, further analysis would be required to see what makes the World Series champions, which is the ultimate goal of all MLB teams.

In conclusion, it's very hard to say exactly how many proportions of team winning percentages can be explained by each aspect of player skills. As I mentioned earlier, baseball is a complicated sport where many external factors affects the ball game like money, weather, or luck. Thus, even if we can build a model with perfect single stats that accurately evaluate each aspect of player skills, such a model would still have some degree of limitations (how would you measure and say what proportions of a team's winning percentage can be explained by the *luck*?). However, I hope this project has given readers some general ideas about what the importance of different aspects of player skills is and how it has changed over time.
