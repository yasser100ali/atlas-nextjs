from openai import OpenAI
from agents import Agent, Runner
import numpy as np
import pandas as pd
import sklearn 

"""
ReAct based data scientist agent 

Newer models are smarter, so a single coding agent could handle more tasks in one given go. 
But now this will be react based so it will look over the last iteration and decide if that is enought to answer
user prompt or if it should keep going. 
"""

# first buid the orchestrator agent 