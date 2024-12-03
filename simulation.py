import datetime as dt
import math
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import random
import seaborn as sns
import time

from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from functools import reduce
from itertools import chain
from multiprocessing import Pool
from tqdm import tqdm
from typing import Any, List, TypeVar, Generic

plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preamble'] = r'\usepackage{lmodern}'

Location = TypeVar("Location")

@dataclass
class Individual(Generic[Location]):
  fitness: float
  deme: Location

class Event(Enum):
  BIRTH = 'birth'
  DEATH = 'death'


def simulate(
  G: nx.DiGraph,
  init_n: int,
  C: int,
  t_end: pd.Timedelta,
  migration_amount: int,
  migration_interval: pd.Timedelta,
  death_rate: float = 1,
  fitness_variance: float = .1,
):
  """
  Simulate evolutionary dynamics in a deme structured population.

  :params:
    - G:     the structure of the demes.
    - C:     the capacity of each deme.
    - t_end: the end time of the simulation.
    - migration_interval: the interval (in days) at which to migrate individuals.
    - death_rate: the death rate.
    - fitness_variance: the variance of the fitness.
  """
  assert init_n <= C, "Initial deme population size cannot exceed capacity."
  assert migration_amount <= C, "Migration amount cannot exceed capacity."
  keyed_data = defaultdict(list)

  def migrate():
    if len(G) == 1: return

    buffers = {deme: [] for deme in G.nodes}
    deme_to_individuals = {deme: [individual for individual in individuals if individual.deme == deme] for deme in G.nodes}
    for from_deme, to_deme in G.edges:
      individuals_in_deme = deme_to_individuals[from_deme]
      individuals_to_migrate = random.sample(individuals_in_deme, k=min(migration_amount, len(individuals_in_deme)))
      for individual in individuals_to_migrate:
        individuals_in_deme.remove(individual)
        buffers[to_deme].append(individual)

    for new_deme, migrated_individuals in buffers.items():
      for individual in migrated_individuals:
        individual.deme = new_deme

  def append_data():
    deme_to_individuals = {deme: [individual for individual in individuals if individual.deme == deme] for deme in G.nodes}
    for deme, individuals_in_deme in deme_to_individuals.items():
      keyed_data[deme].append((t, len(individuals_in_deme), np.average([i.fitness for i in individuals_in_deme]) if individuals_in_deme else np.NaN))
    keyed_data['total'].append((t, len(individuals), np.average([i.fitness for i in individuals]) if individuals else np.NaN))

  # Step 1: Initialization.
  t = pd.Timedelta(0)
  individuals: List[Individual] = [
    Individual(fitness=1.0, deme=deme)
    for deme in G.nodes
    for _ in range(init_n)
  ]
  noise = lambda: np.random.normal(0, fitness_variance)

  last_migration_time = pd.Timedelta(0)
  append_data()
  while t < t_end and individuals:
    # Migrate.
    while t - last_migration_time >= migration_interval:
      migrate()
      last_migration_time += migration_interval

    # Step 2: .
    weights = list(chain(*[
      [individual.fitness, death_rate]
      for individual in individuals
    ]))

    population = list(chain(*[
      [(Event.BIRTH, individual), (Event.DEATH, individual)]
      for individual in individuals
    ]))

    # Step 3: Pick reaction.
    event, individual = random.choices(population, weights=weights)[0]

    # Step 4a: Update population.
    if event == Event.BIRTH:
      new_fitness = individual.fitness + noise()
      new_individual = Individual(fitness=new_fitness, deme=individual.deme)

      current_deme_population_size = len([individual for individual in individuals if individual.deme == new_individual.deme])
      if current_deme_population_size < C and new_fitness >= 0:
        individuals.append(new_individual)
    elif event == Event.DEATH:
      individuals.remove(individual)
    else:
      raise ValueError(f"Unknown event: {event}")

    # Step 4b: Update time.
    r1 = np.random.uniform(0, 1)
    total_weight = sum(weights)
    tau = (1/total_weight) * np.log(1/r1)

    t += pd.Timedelta(days=tau)

    # Logging.
    append_data()

  # set return
  ret = {
    deme: pd.DataFrame(data, columns=['t', 'population_size', 'average_fitness'])
    for deme, data in keyed_data.items()
  }
  for deme, df in ret.items():
    df.set_index('t', inplace=True)

  return ret

C = 60
t_end = pd.Timedelta(days=30)
migration_amount = 10
migration_interval = pd.Timedelta(days=1)
death_rate = .25
fitness_variance = .1
trials = 100
time_granularity = pd.Timedelta('1h')

def simulation_cut(trial: int):
  mas = [int(x) for x in np.linspace(0, C, num=9, endpoint=True)]
  df_dicts = [
    simulate(
      G=nx.from_edgelist([('1', '2')], create_using=nx.DiGraph),
      init_n=C,
      C=C,
      t_end=t_end,
      migration_amount=ma,
      migration_interval=migration_interval,
      death_rate=death_rate,
      fitness_variance=fitness_variance,
    )
    for ma in mas
  ]

  for df_dict, network in zip(df_dicts, [f'ma={ma}' for ma in mas]):
    for deme in df_dict:
      df_dict[deme] = df_dict[deme].resample(time_granularity).ffill()
      df_dict[deme]['deme'] = deme
      df_dict[deme]['trial'] = trial
      df_dict[deme]['network'] = network
      df_dict[deme].reset_index(inplace=True)

  return pd.concat(chain(*[df_dict.values() for df_dict in df_dicts]), ignore_index=True)



def simulations(trial: int):
  df_dicts = [
    simulate(
      G=nx.complete_graph(['1']),
      init_n=C,
      C=C,
      t_end=t_end,
      migration_amount=migration_amount,
      migration_interval=migration_interval,
      death_rate=death_rate,
      fitness_variance=fitness_variance,
    ),
    simulate(
      G=nx.complete_graph(['1']),
      init_n=C,
      C=3*C,
      t_end=t_end,
      migration_amount=migration_amount,
      migration_interval=migration_interval,
      death_rate=death_rate,
      fitness_variance=fitness_variance,
    ),
    simulate(
      G=nx.from_edgelist([('1', '2'), ('2', '1')], create_using=nx.DiGraph),
      init_n=C // 2,
      C=C,
      t_end=t_end,
      migration_amount=migration_amount,
      migration_interval=migration_interval,
      death_rate=death_rate,
      fitness_variance=fitness_variance,
    ),
    simulate(
      G=nx.from_edgelist([('1', '2')], create_using=nx.DiGraph),
      init_n=C // 2,
      C=C,
      t_end=t_end,
      migration_amount=migration_amount,
      migration_interval=migration_interval,
      death_rate=death_rate,
      fitness_variance=fitness_variance,
    ),
    simulate(
      G=nx.from_edgelist([('1', '2'), ('2', '1'), ('2', '3'), ('3', '2'), ('3', '1'), ('1', '3')], create_using=nx.DiGraph),
      init_n=C // 3,
      C=C,
      t_end=t_end,
      migration_amount=migration_amount,
      migration_interval=migration_interval,
      death_rate=death_rate,
      fitness_variance=fitness_variance,
    ),
    simulate(
      G=nx.from_edgelist([('1', '2'), ('2', '3')], create_using=nx.DiGraph),
      init_n=C // 3,
      C=C,
      t_end=t_end,
      migration_amount=migration_amount,
      migration_interval=migration_interval,
      death_rate=death_rate,
      fitness_variance=fitness_variance,
    ),
    simulate(
      G=nx.from_edgelist([('1', '2'), ('2', '3'), ('3', '1')], create_using=nx.DiGraph),
      init_n=C // 3,
      C=C,
      t_end=t_end,
      migration_amount=migration_amount,
      migration_interval=migration_interval,
      death_rate=death_rate,
      fitness_variance=fitness_variance,
    ),
    simulate(
      G=nx.from_edgelist([('3', '1'), ('3', '2')], create_using=nx.DiGraph),
      init_n=C // 3,
      C=C,
      t_end=t_end,
      migration_amount=migration_amount,
      migration_interval=migration_interval,
      death_rate=death_rate,
      fitness_variance=fitness_variance,
    ),
    simulate(
      G=nx.from_edgelist([('1', '3'), ('2', '3')], create_using=nx.DiGraph),
      init_n=C // 3,
      C=C,
      t_end=t_end,
      migration_amount=migration_amount,
      migration_interval=migration_interval,
      death_rate=death_rate,
      fitness_variance=fitness_variance,
    ),
  ]

  for df_dict, network in zip(df_dicts, ['complete n=1', 'complete n=1 w/ 3C cap', 'complete n=2', 'directed path n=2', 'complete n=3', 'directed path n=3', 'directed cycle n=3', 'burst n=3', 'inverse burst n=3']):
    for deme in df_dict:
      df_dict[deme] = df_dict[deme].resample(time_granularity).ffill()
      df_dict[deme]['deme'] = deme
      df_dict[deme]['trial'] = trial
      df_dict[deme]['network'] = network
      df_dict[deme].reset_index(inplace=True)

  return pd.concat(chain(*[df_dict.values() for df_dict in df_dicts]), ignore_index=True)



if __name__ == '__main__':
  with Pool(32) as pool:
    dfs = pool.map(simulation_cut, tqdm(range(trials)))

  df = pd.concat(dfs, ignore_index=True)
  df['time_s'] = df['t'].dt.total_seconds()
  df['time_d'] = df['t'].dt.total_seconds() / 86_400
  df.to_csv(f'data/mi-{int(time.time())}.csv')