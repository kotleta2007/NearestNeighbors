import os
from dotenv import load_dotenv
from llama_index.llms.groq import Groq
import json
from typing import Callable, Dict, Any
import pandas as pd
import matplotlib.pyplot as plt
import time
import numpy as np
import duckdb
from utils.column_names import clean_column_names
import plotly.graph_objects as go

class LLMRouter:
    def __init__(self, model: str = "mixtral-8x7b-32768"):
        load_dotenv()
        self.llm = Groq(api_key=os.getenv('GROQ_API_KEY'), model=model)
        self.function_map: Dict[str, Callable] = {}
        # self.data = pd.read_csv("files/BRISTOR_Zegoland.csv")
        self.data = pd.read_csv("files/preds.csv")

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

Functions with their parameters:

load_data:
- file_paths: list of files to load

modify_data:
- operation: 'increase' or 'decrease'
- timeframe: when the change happens
- amount: change in percent
- column: column to modify

train_model:
no parameters

answer_text:
- string that contains the question the user asked

answer_visual:
- plot_type: type of visualization
- x: x-axis column
- y: list of y-axis columns

END FUNCTIONS

User query: {query}

Return a JSON with:
- function: The function to call
- reasoning: Why this function was chosen
- parameters: parameters matching the schema for that function

Output JSON only."""

        response = self.llm.complete(prompt)
        return json.loads(response.text)

    def route_and_execute(self, query: str) -> Any:
        classification = self.classify_query(query)
        function_name = classification['function']

        if function_name not in self.function_map:
            raise ValueError(f"Function {function_name} not registered")

        return self.function_map[function_name](classification['parameters'])

def load_data(query: str) -> str:
    try:
        import streamlit as st
        import pandas as pd
        import os
        import ast

        # Parse query string to dict
        query_dict = ast.literal_eval(query) if isinstance(query, str) else query

        # Get first filename from file_paths list
        filename = query_dict['file_paths'][0]
        filepath = os.path.join("files", filename)

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File {filepath} not found")

        # Load data from CSV
        df = pd.read_csv(filepath)

        # Store in router's state
        st.session_state.router.data = df

        return f"Loaded {filename}: {df.shape[0]} rows, {df.shape[1]} columns"
    except Exception as e:
        raise Exception(f"Error loading data: {str(e)}")

def modify_data(params):
    print(params)
    mock_df = pd.DataFrame({'date': pd.date_range('2024-01-01', periods=12, freq='M'),
                           'saeles': [100] * 12})
    return mock_df

def train_model(params):
    # skip for now
    print(params)
    time.sleep(5)
    return "Model trained successfully"

def answer_text(query: str) -> str:
    import streamlit as st
    print(st.session_state.router)
    df = clean_column_names(st.session_state.router.data)
    print("Answer in text called")
    print("Query: ", query)
    try:
        load_dotenv()
        llm = Groq(api_key=os.getenv('GROQ_API_KEY'), model="mixtral-8x7b-32768")

        # Build column descriptions
        column_descriptions = []
        for col in df.columns:
            dtype = str(df[col].dtype)
            sample = str(df[col].iloc[0])
            description = f"- {col}: type {dtype}, example value: {sample}"
            column_descriptions.append(description)

        prompt = f"""Given a DataFrame with columns:
{chr(10).join(column_descriptions)}

Convert this user question to a SQL query that will answer it.
Only return the SQL query, nothing else.

The DataFrame is called 'df'.
Use standard SQL syntax compatible with DuckDB.

User question: {query}

Example SQL queries:
- "What's the average {df.columns[1]}?" → SELECT AVG({df.columns[1]}) as value FROM df
- "Show monthly trends" → SELECT strftime(date, '%Y-%m') as month, AVG({df.columns[1]}) as avg_{df.columns[1]} FROM df GROUP BY month ORDER BY month
- "What's the highest {df.columns[1]}?" → SELECT MAX({df.columns[1]}) as value FROM df

SQL query:"""

        # Get and execute SQL query
        response = llm.complete(prompt)
        sql = response.text.strip()

        print(f"SQL Query: {sql}")

        # Execute query
        con = duckdb.connect()
        con.register('df', df)
        result = con.execute(sql).fetchdf()
        con.close()

        # Format response
        if 'month' in result.columns:
            trends = result.to_dict('records')
            return f"Monthly trends:\n" + \
                   "\n".join([f"{row['month']}: {row['avg_' + df.columns[1]]:.2f}" for row in trends])
        elif 'value' in result.columns:
            value = result['value'].iloc[0]
            return f"The result is {value}"
            # value = result['value'].iloc[0]
            # return f"The result is {value:.2f} units"
        else:
            return str(result)

    except Exception as e:
        raise e

def answer_visual(query: str) -> go.Figure:
    try:
        import streamlit as st
        import json
        import re
        df = clean_column_names(st.session_state.router.data)

        load_dotenv()
        llm = Groq(api_key=os.getenv('GROQ_API_KEY'), model="mixtral-8x7b-32768")

        prompt = f"""Given a DataFrame with columns:
{chr(10).join([f"- {col}: {df[col].dtype}" for col in df.columns])}

Convert this visualization request to plot parameters.
Return a JSON object with these fields:
- plot_type: (line, bar, scatter)
- x: column name for x-axis
- y: list of column names for y-axis
- title: appropriate title for the plot

User request: {query}

Example outputs:
{{"plot_type": "line", "x": "date", "y": ["consumption"], "title": "Consumption Over Time"}}
{{"plot_type": "bar", "x": "month", "y": ["total_sales"], "title": "Monthly Sales Distribution"}}

JSON object:"""

        # Get plot parameters from LLM
        response = llm.complete(prompt)
        print("RESPONSE:")
        print(response)

        # Extract JSON from response
        try:
            # First try to find JSON pattern in the response
            json_match = re.search(r'\{.*\}', response.text, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                params = json.loads(json_str)
            else:
                # If no JSON pattern found, try to parse the whole response
                params = json.loads(response.text.strip())
        except json.JSONDecodeError:
            raise ValueError("Could not parse JSON from model response")

        print("PARAMS:")
        print(params)

        # Create figure based on plot type
        fig = go.Figure()

        if params['plot_type'] == 'line':
            for y_col in params['y']:
                fig.add_trace(go.Scatter(
                    x=df[params['x']],
                    y=df[y_col],
                    mode='lines+markers',
                    name=y_col
                ))

        elif params['plot_type'] == 'bar':
            for y_col in params['y']:
                fig.add_trace(go.Bar(
                    x=df[params['x']],
                    y=df[y_col],
                    name=y_col
                ))

        elif params['plot_type'] == 'scatter':
            for y_col in params['y']:
                fig.add_trace(go.Scatter(
                    x=df[params['x']],
                    y=df[y_col],
                    mode='markers',
                    name=y_col
                ))

        # Update layout
        fig.update_layout(
            title=params['title'],
            xaxis_title=params['x'],
            yaxis_title=', '.join(params['y']),
            template="plotly_white",
            showlegend=len(params['y']) > 1
        )

        return fig

    except Exception as e:
        raise e

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
