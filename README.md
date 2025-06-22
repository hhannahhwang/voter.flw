# AIhacks

design : [figma file](https://www.figma.com/design/MrwOplFUpVYOCeHn1Ki35T/aihacks?node-id=0-1&p=f&t=VoieXtVuW1p85li8-0)

## Inspiration
The voting systems that uphold the foundation of our republic, its policy, and its power are among the most sacred tools that we, the people, have in order to ensure our representation in our government. But what happens when those very systems are corrupted, or otherwise tampered with? In light of recent lawsuits regarding possible voting irregularities in New York, an observed increase in voter suppression efforts since 2020, and 79 restrictive voter laws being passed from 2021 to 2024, we wanted to analyze an area that implemented these policies prior to the 2024 election. We placed a focus on analyzing how big of an impact these policies actually had on voter participation through the use of various demographic statistics.  

## What it does
Using Flower.ai, we implemented a federated learning-based model in order to accurately predict voter results based on past voting trends and demographic data. If there were any significant discrepancies in expected results and actual data, or there were demographic variables that pointed towards possible risk of suppression, it was flagged as "high-risk" of possibly voter-suppression activities. 

The output of our federated learning (FL) model includes a county-specific risk score, represented as a decimal between 0 and 1. A lower score (closer to 0) indicates a higher risk of voter suppression, while a higher score (closer to 1) indicates lower risk. Additionally, the model uses SHAP analysis to identify the 10 demographic variables that most heavily influenced each county's risk score, along with each variable’s corresponding weight.

We then ran this raw data through an AI wrapper, implementing Google's Gemini API to cross-reference voting policy changes with counties that were flagged by our model as being "high risk" of possible voter suppression. Using compiled news articles outlining voter policy changes since 2022, in addition to the demographic variables that most heavily contributed to the risk score calculation, Gemini generates a sophisticated explanation as to what is most likely to cause voter suppression. These justifications for higher risk scores included a focus on groups that could be at risk of being targeted by voter suppression, and other policies like underfunding, gerrymandering, and an increases in incidents regarding partisan poll-watchers.

After the data was provided justifiable causes by the Gemini wrapper, it was then aggregated into a interactive map visualization using front-end frameworks like TailwindCSS, Next.js, and React, in order to accurately display county data, their risk scores, and the justifications for each risk score. Hovering over each individual county would display its information, and an organized sidebar with a search feature allows the user to easily find any specific county in the state.

## How we built it
We initially started by scouring the web, trying to find a dataset. We eventually found a North Carolina collection of election data, https://www.ncsbe.gov/results-data, which gave us access to a county-by-county basis of individuals, the party they voted for, their demographics, and their method of voting.

We then spent time aggregating data, trying to create a clean efficient dataset to predict from. Afterwards, we used scikit-learn to create an initial Random Forest Classifier, aiming to predict whether a person should, if no outside factors were to change, vote in the 2024 election. Once we began to get favorable results on a single county, we made a federated learning loop.

With the current version of the project, we had nearly 100 separate counties, each of varying sizes. As we hope to later expand, we would need to manage over 3000 counties, some of which are incredibly large. Holding all of this data in one place is quite unrealistic, especially if we wanted to continue training our model on older and older data. As a result, since each county already stores their own data, we found it easier to use federated learning, sending the model to each county, instead of having to combine an enormous, and unrealistic for us, dataset.

We then trained and saved a model on each county, using Flower.ai. Afterwards, we used that combined model, computing each individual person's likelihood to vote, compared that to each individual's actual chance of voting, and computed a mean value for each county, discovering how "off" they were from what we anticipated. We then used this to create a uniform distributed risk factor, noticing that when places tended to be in risk of having a lower score than they should.

We then did excessive research into specific North Carolina polling changes, and aggregated this information with the SHAP analysis from each training loop, piecing together the most likely reasons for each county to be at risk. Combined, we have the relevant data to flag areas as potentially experiencing voter suppression, and the likely reasons for this to occur, allowing government officials to provide a quick and simple solution.

To verify that this information was accurate, we found that actual data of voter turnout, per county, in 2020 and 2024, and calculated a percentage of increase or decrease in voters. Obviously, there are millions of factors that will influence the difference in numbers between these two times. But our goal is not to find the exact counties that will experience change, but those that are most at risk for potential suppression. As a result, though we do have some false negatives/positives, in correlation to that exact data, it is simply a trend, not our entire model. See the second image below for the described graph.

## Challenges we ran into
One of the main challenges we ran into was trying to integrate an LLM API into the wrapper portion of the project. At first, the project was setup to use Anthropic's Claude API to generate our explanations based off of policy changes and the model data, but we realized that we hadn't pre-registered for the trial credits that were offered by Anthropic for the purpose of this hackathon. After refactoring the code to use OpenAi's GPT-3.5 instead, we ran into a similar issue, in that OpenAI discontinued their program that offered trial API credits to first-time users. In the end, we ended up using Google's Gemini API, as it was free (with the trade-off of being rate-limited). To get around the rate limitations, we just ran our analysis script multiple times, skipping counties that had already been covered by the previous iterations.

We also had some challenges implementing the map visualization, as we had to find an svg, transform it into an interactive map by identifying the paths of each separate county boundary, and dealing with animations. Having to deal with functionality on top of aesthetic made the front-end development portion a bit more challenging than anticipated.  

## Accomplishments that we're proud of
This was our first introduction to federated learning, so being able to implement our idea using it, learning how the model worked, and tweaking it until the results were accurate to real data was quite fulfilling. Additionally, this was our first real experience with making our own model. We found our data, preprocessed it, and tweaked hyperparameters of varying models. We fully made our own design, and were able to use it exactly how we wanted.

## What we learned
The main thing that we learned during this project was about the concept of federated learning. Prior to coming here, we had never heard of the concept, but after spending a day exploring it, we feel it is an essential and under-utilized aspect of machine learning. We were fascinated by how it guaranteed user privacy, allowed for new kinds of optimizations, and how it fit our project design perfectly.

We also had never actually trained our own model completely from scratch before, and we were overjoyed to be able to go through the entire development pipeline completely on our own, and now feel much more prepared to continuing developing and improving models going forward.

## What's next for voter.flw
Right now, due to the data we found, voter.flw only highlights voter suppression for North Carolina. This, though an effective demo, would be the first thing that we would want to resolve, aiming to instead cover a complete national audience.

Another idea that we wanted to continue with was using something such as Apache Kafka to allow for real time additions to our models. Right now, we can use data from 2024 to confirm that voter suppression may have occurred in 2024. With access to real time trends, we can continuously check for voter suppression throughout the entire voting process. Furthermore, due to our federated learning training loop, this is not a violation of any voter privacy, as we will guarantee that any voter information does not leave the county database. This would make our tool an essential part of the entire election cycle, guaranteeing fair and safe access to the polls.
