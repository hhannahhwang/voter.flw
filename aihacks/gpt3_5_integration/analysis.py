import openai
import os

class OpenAIVoterSuppressionAnalyzer:
    def __init__(self, api_key, model="gpt-3.5-turbo"):
        openai.api_key = api_key
        self.model = model

    def build_prompt(self, county, features, articles):
        feature_text = "\n".join([f"{feat}: {importance:.6f}" for feat, importance in features.items()])
        article_text = "\n\n".join([f"Article {i+1}:\n{a}" for i, a in enumerate(articles)])

        return f"""
You are a political analyst.

Using the following top features from {county} county and current news articles, write a short data-informed paragraph explaining how voter behavior in this county might suggest voter suppression patterns.

Top Features:
{feature_text}

News Articles:
{article_text}

Give 1 paragraph that includes demographic or behavioral insights suggesting potential voter suppression tactics or policy impacts.
"""

    def analyze(self, county, features, articles):
        prompt = self.build_prompt(county, features, articles)
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=800,
            temperature=0.7
        )
        return response["choices"][0]["message"]["content"]
