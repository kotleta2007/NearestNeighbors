import os
from dotenv import load_dotenv
from llama_index.llms.groq import Groq
import json
from typing import Callable, Dict, Any
import pandas as pd
import matplotlib.pyplot as plt

class LLMRouter:
    def __init__(self, model: str = "mixtral-8x7b-32768"):
        load_dotenv()
        self.llm = Groq(api_key=os.getenv('GROQ_API_KEY'), model=model)
        self.function_map: Dict[str, Callable] = {}
        self.data = None

    def register_function(self, name: str, func: Callable):
        self.function_map[name] = func

    def classify_query(self, query: str) -> dict:
        prompt = f"""Given the user query, determine which function should be called and extract relevant parameters.
Available functions:
- load_data(): Load data for training
- modify_data(): Transform existing data
- train_model(): Train time series model
- answer_text(): Answer questions about data/predictions with text
- answer_visual(): Answer questions about data/predictions with visualizations

User query: {query}

Return a JSON with:
- function: The function to call
- reasoning: Why this function was chosen
- parameters: Any relevant parameters extracted from query

Output JSON only."""

        response = self.llm.complete(prompt)
        return json.loads(response.text)

    def route_and_execute(self, query: str) -> Any:
        classification = self.classify_query(query)
        function_name = classification['function']

        if function_name not in self.function_map:
            raise ValueError(f"Function {function_name} not registered")

        return self.function_map[function_name](classification['parameters'])

def load_data(params):
    print(params)
    return pd.DataFrame({'date': pd.date_range('2024-01-01', periods=12, freq='M'),
                        'sales': [100, 120, 110, 130, 140, 135, 150, 160, 155, 170, 180, 175]})

def modify_data(params):
    print(params)
    mock_df = pd.DataFrame({'date': pd.date_range('2024-01-01', periods=12, freq='M'),
                           'saeles': [100] * 12})
    return mock_df

def train_model(params):
    print(params)
    return "Model trained successfully"

def answer_text(params):
    print(params)
    return "Average sales are 100 units per month"

def answer_visual(params):
    print(params)
    import plotly.graph_objects as go

    # Example data - replace with actual data processing
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[1, 2, 3], y=[4, 5, 6], mode='lines+markers'))

    fig.update_layout(
        title="Sales Trend",
        xaxis_title="Time",
        yaxis_title="Sales",
        template="plotly_white"
    )

    return fig

# Usage example
if __name__ == "__main__":
    router = LLMRouter()
    router.register_function("load_data", load_data)
    router.register_function("modify_data", modify_data)
    router.register_function("train_model", train_model)
    router.register_function("answer_text", answer_text)
    router.register_function("answer_visual", answer_visual)

    result = router.route_and_execute("Load the sales data from CSV")
    print(result)
