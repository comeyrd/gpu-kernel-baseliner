from typing import Optional,TypeAlias,Any
from collections import defaultdict
import copy
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

def merge_options_maps(*maps: OptionsMap) -> OptionsMap:
    merged: OptionsMap = defaultdict(dict)
    for m in maps:
        for interface_name, in_dict in m.items():
            for option_name, val in in_dict.items():
                merged[interface_name][option_name] = val
    return dict(merged)

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
  benchmark:PlannedComponent
  stats:PlannedStat
  sweep:Optional[SweepSpec] = None

### Reports
class Metric(BaseModel):
  name:str
  unit:str
  data:Any

class RunReport(BaseModel):
  id:str
  sweep_point:OptionsMap
  measurements:list[Metric]
  
class Hardware(BaseModel):
  card_name:str
  
class BenchmarkReport(BaseModel):
  id:str
  results:list[RunReport]
  hardware:Hardware

class BenchmarkExecution(BaseModel):
  plan:BenchmarkPlan
  benchmark_report:BenchmarkReport

class CampaignReport(BaseModel):
  id:str
  name:str
  recipe_name:str
  recipe:Recipe       #[backend,[workload,Exec]
  benchmark_runs:dict[str,dict[str,BenchmarkExecution]]
    
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
  m_original_json:json
  m_report:Report
  m_vectors_df:pd.DataFrame
  m_scalars_df:pd.DataFrame
  m_metadata_df:pd.DataFrame # datetime,baseliner_version,git_version,report_id,campaign_id,benchmark_id,run_id,interface.option
  
  def __init__(self,json_filepath):
    self.m_original_json = load_json(json_filepath)
    self.m_report =  Report.model_validate(self.m_original_json) 
    self.populate_dataframes()
    
  def filter(self, **kwargs) -> "ReportDataframe":
    mask = pd.Series(True, index=self.m_metadata_df.index)
    
    for key, value in kwargs.items():
        if key not in self.m_metadata_df.columns:
            available = self.m_metadata_df.columns.tolist()
            raise ValueError(f"Column '{key}' not found. Available: {available}")
        
        if isinstance(value, list):
            mask &= self.m_metadata_df[key].isin(value)
        else:
            mask &= self.m_metadata_df[key] == value
    
    filtered_run_ids = self.m_metadata_df[mask]["run_id"]
    
    result = copy.copy(self)
    result.m_metadata_df = self.m_metadata_df[mask].reset_index(drop=True)
    result.m_scalars_df = self.m_scalars_df[
        self.m_scalars_df["run_id"].isin(filtered_run_ids)
    ].reset_index(drop=True)
    result.m_vectors_df = self.m_vectors_df[
        self.m_vectors_df["run_id"].isin(filtered_run_ids)
    ].reset_index(drop=True)
    
    return result

  def to_csv(self,folder):
    
    with open(folder+"/report.json","w") as f:
      json.dump(self.m_original_json,f,indent=2)
    self.m_metadata_df.to_csv(folder+"/metadata.csv")
    self.m_scalars_df.to_csv(folder+"/scalars.csv")
    self.m_vectors_df.to_csv(folder+"/vectors.csv")
    
    
  def populate_dataframes(self):
      metadata_rows = []
      vector_rows = defaultdict(dict)
      scalar_rows = defaultdict(dict)
      
      for campaign in self.m_report.campaign_runs:
        for backend, inner_dict in campaign.benchmark_runs.items():
          for workload, bench_exec in inner_dict.items():
            
            full_options: OptionsMap = defaultdict(dict)
            full_options = merge_options_maps(
                full_options,
                bench_exec.plan.backend.options,
                bench_exec.plan.workload.options,
                bench_exec.plan.stopping.options,
                bench_exec.plan.stats.options,
                bench_exec.plan.benchmark.options
            )
            
            for run in bench_exec.benchmark_report.results:
              row_data = {
                  "datetime": self.m_report.datetime,
                  "baseliner_version": self.m_report.baseliner_version,
                  "git_version": self.m_report.git_version,
                  "report_id": self.m_report.id,
                  "campaign_id": campaign.id,
                  "benchmark_id": bench_exec.benchmark_report.id,
                  "run_id": run.id,
                  "backend": bench_exec.plan.backend.impl,
                  "workload": bench_exec.plan.workload.impl,
                  "stopping": bench_exec.plan.stopping.impl,
                  "hardware.card_name": bench_exec.benchmark_report.hardware.card_name
              }

              options: OptionsMap = merge_options_maps(full_options, run.sweep_point)
              for interface, in_dict in options.items():
                for option, inner in in_dict.items():
                  row_data[f"{interface}.{option}"] = inner.value
              
              metadata_rows.append(row_data)

              for metrics in run.measurements:
                if isinstance(metrics.data, list):
                  for i, val in enumerate(metrics.data):
                      vector_rows[(run.id, i)]["run_id"] = run.id
                      vector_rows[(run.id, i)]["run_nb"] = i
                      vector_rows[(run.id, i)][metrics.name] = val
                      if metrics.unit != "":
                        vector_rows[(run.id, i)][f"{metrics.name}.unit"] = metrics.unit
                else:
                  scalar_rows[run.id]["run_id"] = run.id
                  scalar_rows[run.id][metrics.name] = metrics.data
                  scalar_rows[run.id][f"{metrics.name}.unit"] = metrics.unit
                  
      self.m_metadata_df = pd.DataFrame(metadata_rows)
      self.m_scalars_df = pd.DataFrame(scalar_rows.values())
      self.m_vectors_df = pd.DataFrame(vector_rows.values())
