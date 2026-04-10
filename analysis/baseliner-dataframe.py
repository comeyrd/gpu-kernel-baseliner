from typing import Optional,TypeAlias,Any
from enum import Enum
import pandas as pd
import json 
from dataclasses import dataclass
##TODO USE PYDANTIC for serialization deser
##Sweep

@dataclass
class SweepStrategy(Enum):
  FullGrid = 1
  
@dataclass
class SweepPolicy(Enum):
  PowersOfTwo = 1
  LinearRange = 2
  Enumerated = 3
  
  
@dataclass
class SweepHint:
  policy:SweepPolicy
  min:str
  max:str
  step:str
  enumerated:list[str]
  
@dataclass
class SweepAxis:
  interface:str
  option:str
  hint:Optional[SweepHint]
  
  
@dataclass
class SweepSpec:
  strategy:SweepStrategy
  axes:list[SweepAxis]


#### Recipe
@dataclass
class RecipeStat:
  preset:str
  
@dataclass
class RecipeComponent:
  impl:str
  preset:Optional[str]

@dataclass
class Recipe:
  description:str
  benchmark:Optional[RecipeComponent]
  stopping:Optional[RecipeComponent]
  stats:Optional[RecipeStat]
  sweep:Optional[SweepSpec]

### Options

@dataclass
class Option:
  description:Optional[str]
  value:str

OptionsMap:TypeAlias = dict[str,dict[str,Option]]


### Plan

@dataclass 
class PlannedComponent:
  impl:str
  preset:str
  options:OptionsMap

@dataclass
class PlannedStat:
  preset:str
  stats:list[str]
  options:OptionsMap
  

@dataclass 
class BenchmarkPlan:
  workload:PlannedComponent
  backend:PlannedComponent
  stopping:PlannedComponent
  stats:PlannedStat
  sweep:Optional[SweepSpec]

### Reports
@dataclass
class Metric:
  name:str
  unit:str
  data:Any

@dataclass
class SingleRunReport:
  sweep_point:OptionsMap
  measurement:list[Metric]
  
@dataclass
class Hardware:
  name:str
  
@dataclass 
class BenchmarkReport:
  results:list[SingleRunReport]
  hardware:Hardware

@dataclass
class RunReport:
  plan:BenchmarkPlan
  benchmark_report:BenchmarkReport

@dataclass
class CampaignReport:
  name:str
  recipe_name:str
  recipe:Recipe
  benchmark_runs:dict[str,dict[str,RunReport]]
    

@dataclass
class Report:
  """Class to hold a Baseliner Report"""
  baseliner_version:str
  git_version:str
  datetime:str
  campaigns:list[CampaignReport]
  

def load_json(json_filepath):
   with open('data.json', 'r') as file:
    return json.load(file)

def load_baseliner_report(json_filepath):
  json = load_json(json_filepath)
  

