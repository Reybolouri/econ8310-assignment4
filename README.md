# Assignment 4
## Econ 8310 - Business Forecasting

This assignment will make use of the bayesian statistical models covered in Lessons 10 to 12. 

A/B Testing is a critical concept in data science, and for many companies one of the most relevant applications of data-driven decision-making. In order to improve product offerings, marketing campaigns, user interfaces, and many other user-facing interactions, scientists and engineers create experiments to determine the efficacy of proposed changes. Users are then randomly assigned to either the treatment or control group, and their behavior is recorded.
If the changes that the treatment group is exposed to can be measured to have a benefit in the metric of interest, then those changes are scaled up and rolled out to across all interactions.
Below is a short video detailing the A/B Testing process, in case you want to learn a bit more:
[https://youtu.be/DUNk4GPZ9bw](https://youtu.be/DUNk4GPZ9bw)

For this assignment, you will use an A/B test data set, which was pulled from the Kaggle website (https://www.kaggle.com/datasets/yufengsui/mobile-games-ab-testing). I have added the data from the page into Codio for you. It can be found in the cookie_cats.csv file in the file tree. It can also be found at [https://github.com/dustywhite7/Econ8310/raw/master/AssignmentData/cookie_cats.csv](https://github.com/dustywhite7/Econ8310/raw/master/AssignmentData/cookie_cats.csv)

The variables are defined as follows:

| Variable Name  | Definition |
|----------------|----|
| userid         | A unique number that identifies each player  |
| version        | Whether the player was put in the control group (gate_30 - a gate at level 30) or the group with the moved gate (gate_40 - a gate at level 40) |
| sum_gamerounds | The number of game rounds played by the player during the first 14 days after install.  |
| retention1     | Did the player come back and play 1 day after installing?     |
| retention7     | Did the player come back and play 7 days after installing?    |               

### The questions

You will be asked to answer the following questions in a small quiz on Canvas:
1. What was the effect of moving the gate from level 30 to level 40 on 1-day retention rates?
2. What was the effect of moving the gate from level 30 to level 40 on 7-day retention rates?
3. What was the biggest challenge for you in completing this assignment?

You will also be asked to submit a URL to your forked GitHub repository containing your code used to answer these questions.


My Answers:

1. Moving the gate from level 30 to level 40 appears to have slightly decreased 1Day retention by about 0.6 percentage point, and as we can see in the posterior distribution p30 and p40 overlap heavily.so there is little chance it improved the players retention.:  
── 1-DAY retention posterior summary ──
        mean     sd  hdi_2.5%  hdi_97.5%  mcse_mean  mcse_sd  ess_bulk  \
p_30   0.448  0.002     0.444      0.453        0.0      0.0    4330.0   
p_40   0.442  0.002     0.438      0.447        0.0      0.0    4242.0   
delta -0.006  0.003    -0.012      0.001        0.0      0.0    4194.0   

       ess_tail  r_hat  
p_30     2924.0    1.0  
p_40     2763.0    1.0  
delta    2924.0    1.0   

Pr(Δ₁ > 0) = 0.037



2. For 7 day retention, the p30 vs p40 curves sit at about 19 % vs. 18.2%, with almost no overlap.The posterior is entirely below zero (approximately -1.3 to -0.3pp) resulting in a posterior mean of -0.8 pp , so pushing the gate to level 40 reduced seven day retention by roughly 0.8 percent.:

── 7-DAY retention posterior summary ──
        mean     sd  hdi_2.5%  hdi_97.5%  mcse_mean  mcse_sd  ess_bulk  \
p_30   0.190  0.002     0.186      0.194        0.0      0.0    4735.0   
p_40   0.182  0.002     0.178      0.185        0.0      0.0    3401.0   
delta -0.008  0.003    -0.013     -0.003        0.0      0.0    4333.0   

       ess_tail  r_hat  
p_30     3228.0    1.0  
p_40     2866.0    1.0  
delta    3229.0    1.0   

Pr(Δ₇ > 0) = 0.001

3. The biggest challenge for me was underestanding the concept of A&B baysian testing and building the model and also interpreting the results.
