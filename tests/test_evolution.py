"""Tests für Individuum, Population, Evolution und Experiment-Statistiken.

Hinweis zu "Mario-Tempo": Diese Tests prüfen die evolutionsseitige
Mathematik (Selektion, Mutation, Elitismus, Statistiken) mit kleinen,
handprüfbaren Werten - ohne Emulator (der Emulator wird separat
integrativ getestet, siehe test_integration.py).
"""
import numpy as np
import pytest

from mario_evolution.evolution import (EvolutionConfig, Evolver,
                                       rank_individuals, select_parent,
                                       mutate_genome)
from mario_evolution.individual import Individual
from mario_evolution.population import Population
from mario_evolution.experiment import Experiment, GenerationStats


def make_pop(size=4, hidden=(4,), seed=0) -> Population:
    rng = np.random.default_rng(seed)
    return Population.random(size=size, hidden_sizes=list(hidden), rng=rng)


def assign_fitness(pop: Population, values: list[float]) -> None:
    for ind, v in zip(pop.individuals, values):
        ind.fitness = v


class TestIndividualAndPopulation:
    def test_reproducible_initialization(self):
        p1 = make_pop(seed=42)
        p2 = make_pop(seed=42)
        for a, b in zip(p1.individuals, p2.individuals):
            assert np.allclose(a.genome(), b.genome())

    def test_layer_sizes_fixed_to_interface(self):
        from mario_evolution.observation import OBSERVATION_SIZE
        from mario_evolution.actions import ACTION_COUNT
        pop = make_pop()
        assert pop.layer_sizes == [OBSERVATION_SIZE, 4, ACTION_COUNT]

    def test_mixed_genome_lengths_rejected(self):
        rng = np.random.default_rng(0)
        big = Individual.random(hidden_sizes=[8], rng=rng)
        small = Individual.random(hidden_sizes=[4], rng=rng)
        with pytest.raises(ValueError, match="Genomlänge"):
            Population(individuals=[big, small], hidden_sizes=[8])

    def test_invalid_population_size(self):
        with pytest.raises(ValueError):
            Population.random(size=0, hidden_sizes=[4], rng=np.random.default_rng(0))


class TestSelection:
    def test_ranking_best_first(self):
        pop = make_pop()
        assign_fitness(pop, [10.0, 30.0, 20.0, 0.0])
        ranked = rank_individuals(pop.individuals)
        assert ranked[0].fitness == 30.0
        assert ranked[-1].fitness == 0.0

    def test_unrated_individuals_rank_last(self):
        pop = make_pop()
        pop.individuals[0].fitness = 5.0
        ranked = rank_individuals(pop.individuals)
        assert ranked[-1].fitness is None

    def test_best_fitness_gets_chosen_more_often(self):
        """Roulette: Bestindividuum wird statistisch häufiger gewählt."""
        pop = make_pop()
        assign_fitness(pop, [100.0, 1.0, 1.0, 1.0])
        ranked = rank_individuals(pop.individuals)
        rng = np.random.default_rng(7)
        picks = [select_parent(ranked, rng).label for _ in range(500)]
        best_label = ranked[0].label
        assert picks.count(best_label) > 300   # Wahrscheinlichkeit ~100/103

    def test_all_zero_fitness_fair_choice(self):
        pop = make_pop()
        assign_fitness(pop, [0.0, 0.0, 0.0, 0.0])
        ranked = rank_individuals(pop.individuals)
        rng = np.random.default_rng(3)
        picks = {select_parent(ranked, rng).label for _ in range(200)}
        assert len(picks) == 4   # alle gleichwahrscheinlich


class TestMutation:
    def test_only_masked_genes_change(self):
        """Bei rate=0 bleibt das Genom bitidentisch (Eliten-Sicherheit)."""
        rng = np.random.default_rng(1)
        cfg = EvolutionConfig(mutation_rate=0.0, mutation_strength=0.5)
        genome = np.arange(10, dtype=float)
        out = mutate_genome(genome, cfg, rng)
        assert np.array_equal(genome, out)

    def test_mutation_changes_exactly_reported_genes(self):
        rng = np.random.default_rng(1)
        cfg = EvolutionConfig(mutation_rate=1.0, mutation_strength=0.5)
        genome = np.zeros(10)
        out = mutate_genome(genome, cfg, rng)
        assert not np.allclose(genome, out)
        assert np.all(np.isfinite(out))

    def test_mutation_does_not_modify_input(self):
        rng = np.random.default_rng(1)
        cfg = EvolutionConfig(mutation_rate=1.0, mutation_strength=0.5)
        genome = np.ones(5)
        before = genome.copy()
        mutate_genome(genome, cfg, rng)
        assert np.array_equal(genome, before)

    def test_invalid_genome_rejected(self):
        rng = np.random.default_rng(1)
        cfg = EvolutionConfig()
        with pytest.raises(ValueError, match="NaN"):
            mutate_genome(np.array([np.nan]), cfg, rng)


class TestEvolver:
    def test_elites_bitidentical(self):
        """Unveränderte Eliten: Top-k Genome erscheinen 1:1 in der neuen Gen."""
        pop = make_pop(size=6, seed=5)
        assign_fitness(pop, [1.0, 5.0, 3.0, 4.0, 2.0, 0.0])
        cfg = EvolutionConfig(elite_count=2, mutation_rate=0.5, mutation_strength=0.5)
        evolver = Evolver(cfg, np.random.default_rng(11))
        fitness = {ind.label: ind.fitness for ind in pop.individuals}
        new = evolver.next_generation(pop, fitness)
        elite_genomes = sorted([ind.genome().tolist() for ind in new.individuals
                                if ind.label.endswith("_elite")])
        best_genomes = sorted([ind.genome().tolist() for ind in
                               rank_individuals(pop.individuals)[:2]])
        assert elite_genomes == best_genomes

    def test_new_generation_same_size_and_shape(self):
        pop = make_pop(size=6, seed=5)
        assign_fitness(pop, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        cfg = EvolutionConfig(elite_count=2)
        evolver = Evolver(cfg, np.random.default_rng(11))
        fitness = {ind.label: ind.fitness for ind in pop.individuals}
        new = evolver.next_generation(pop, fitness)
        assert new.size == pop.size
        assert new.layer_sizes == pop.layer_sizes

    def test_missing_fitness_rejected(self):
        pop = make_pop(size=3)
        cfg = EvolutionConfig(elite_count=1)
        evolver = Evolver(cfg, np.random.default_rng(11))
        with pytest.raises(ValueError, match="Fitness"):
            evolver.next_generation(pop, {"Ind_000": 1.0, "Ind_001": 2.0})


class TestGenerationStats:
    def test_stats_computation_manual(self):
        """Statistiken über eine Generation - alle Werte von Hand prüfbar."""
        stats = GenerationStats(
            generation=2, population_size=4,
            best_fitness=100.0, average_fitness=25.0, worst_fitness=-10.0,
            success_count=1, success_rate=0.25,
            average_steps=500.0, total_blocked_steps=42, best_x_pos=1000.0,
        )
        d = stats.as_dict()
        assert d["best_fitness"] == 100.0
        assert d["success_rate"] == 0.25
        assert GenerationStats.from_dict(d) == stats

    def test_from_dict_missing_field_rejected(self):
        with pytest.raises(ValueError, match="ungültige Daten"):
            GenerationStats.from_dict({"generation": 1})


class TestExperimentRecordAndSave:
    def test_record_strictly_better_only(self):
        """Rekord wird nur bei strikt besserer Fitness ersetzt."""
        exp = Experiment(population=make_pop(size=2, seed=1),
                         evolution_config=EvolutionConfig(),
                         rng=np.random.default_rng(0), max_steps=10)
        # Simuliere zwei Bewertungsrunden direkt über _update_record:
        class FakeEpi:
            pass
        class FakeSim:
            individuals = exp.population.individuals
        sim = FakeSim()
        exp._update_record(sim, {exp.population.individuals[0].label: 100.0,
                                 exp.population.individuals[1].label: 50.0})
        first_genome = exp.best_overall_genome.copy()
        exp._update_record(sim, {exp.population.individuals[0].label: 100.0,
                                 exp.population.individuals[1].label: 99.0})
        # gleiche/gleich gute Fitness -> Rekord unverändert
        assert np.array_equal(exp.best_overall_genome, first_genome)

    def test_save_and_load_roundtrip(self, tmp_path):
        exp = Experiment(population=make_pop(size=3, seed=2),
                         evolution_config=EvolutionConfig(elite_count=1),
                         rng=np.random.default_rng(9), max_steps=100)
        assign_fitness(exp.population, [10.0, 20.0, 30.0])
        # Rekord manuell setzen (Simulation wird hier nicht ausgeführt)
        exp.best_overall_fitness = 30.0
        exp.best_overall_label = exp.population.individuals[2].label
        exp.best_overall_genome = exp.population.individuals[2].genome().copy()
        path = tmp_path / "exp.json"
        exp.save(path)
        loaded = Experiment.load(path, EvolutionConfig(elite_count=1),
                                 np.random.default_rng(9))
        assert loaded.generation == exp.generation
        assert loaded.best_overall_label == exp.best_overall_label
        assert np.allclose(loaded.best_overall_genome, exp.best_overall_genome)
        for a, b in zip(exp.population.individuals, loaded.population.individuals):
            assert a.label == b.label
            assert a.fitness == b.fitness
            assert np.allclose(a.genome(), b.genome())

    def test_advance_requires_fitness(self):
        exp = Experiment(population=make_pop(size=2, seed=3),
                         evolution_config=EvolutionConfig(),
                         rng=np.random.default_rng(0))
        with pytest.raises(RuntimeError, match="nicht bewertet"):
            exp.advance_generation()
