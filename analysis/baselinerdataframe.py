from typing import Optional,TypeAlias,Any
from enum import Enum
import pandas as pd
import json 
from dataclasses import dataclass
from pydantic import BaseModel, Field
##Sweep

class SweepStrategy(str,Enum):
  FullGrid = "FullGrid"
  
class SweepPolicy(str,Enum):
  PowersOfTwo = "PowersOfTwo"
  LinearRange = "LinearRange"
  Enumerated = "Enumerated"
  
  

class SweepHint(BaseModel):
  policy:SweepPolicy
  min:str
  max:str
  step:str
  enumerated:list[str]
  
class SweepAxis(BaseModel):
  interface:str
  option:str
  hint:Optional[SweepHint] = None
  
  

class SweepSpec(BaseModel):
  strategy:SweepStrategy
  axes:list[SweepAxis]


#### Recipe

class RecipeStat(BaseModel):
  preset:str
  

class RecipeComponent(BaseModel):
  impl:str
  preset:Optional[str] = None


class Recipe(BaseModel):
  description:str
  benchmark:Optional[RecipeComponent] = None
  stopping:Optional[RecipeComponent] = None
  stats:Optional[RecipeStat] = None
  sweep:Optional[SweepSpec] = None

### Options


class Option(BaseModel):
  description:Optional[str]=None
  value:str

OptionsMap:TypeAlias = dict[str,dict[str,Option]]


### Plan


class PlannedComponent(BaseModel):
  impl:str
  preset:str
  options:OptionsMap


class PlannedStat(BaseModel):
  preset:str
  stats:list[str]
  options:OptionsMap
  


class BenchmarkPlan(BaseModel):
  workload:PlannedComponent
  backend:PlannedComponent
  stopping:PlannedComponent
  stats:PlannedStat
  sweep:Optional[SweepSpec] = None

### Reports
class Metric(BaseModel):
  name:str
  unit:str
  data:Any

class SingleRunReport(BaseModel):
  id:str
  sweep_point:OptionsMap
  measurements:list[Metric]
  
class Hardware(BaseModel):
  name:str
  
class BenchmarkReport(BaseModel):
  id:str
  results:list[SingleRunReport]
  hardware:Hardware

class RunReport(BaseModel):
  id:str
  plan:BenchmarkPlan
  benchmark_report:BenchmarkReport

class CampaignReport(BaseModel):
  id:str
  name:str
  recipe_name:str
  recipe:Recipe
  benchmark_runs:dict[str,dict[str,RunReport]]
    
class Report(BaseModel):
  """Class to hold a Baseliner Report"""
  baseliner_version:str
  id:str
  git_version:str
  datetime:str
  campaign_runs:list[CampaignReport]
  

def load_json(json_filepath):
   with open(json_filepath, 'r') as file:
    return json.load(file)



class ReportDataframe:
  m_report:Report
  m_vectors_df:pd.DataFrame
  m_scalars_df:pd.DataFrame
  
  def __init__(self,json_filepath):
    json = load_json(json_filepath)
    self.m_report =  Report.model_validate(json) 
  
  