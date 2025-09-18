# build a data analyst agent that delegates tasks to lower agents 

from agents import Agent, Runner

DataAnalystAgent = Agent(
    name="data_analyst_agent",
    model="gpt-4.1"
)