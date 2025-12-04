#!/usr/bin/env python
"""
Example: Running malaria intervention scenarios with MINTe.

This script demonstrates how to use the MINTe package to:
1. Run basic intervention scenarios
2. Compare different net types
3. Visualize results
"""

import numpy as np
import pandas as pd

# Import MINTe components
from minte import (
    run_minter_scenarios,
    run_malaria_emulator,
    create_scenarios,
    create_scenario_plots,
    preload_all_models,
)


def example_basic_scenario():
    """Run a basic single scenario."""
    print("=" * 60)
    print("Example 1: Basic Single Scenario")
    print("=" * 60)

    results = run_minter_scenarios(
        # Resistance level
        res_use=[0.3],
        # Current net coverage
        py_only=[0.4],
        py_pbo=[0.3],
        py_pyrrole=[0.2],
        py_ppf=[0.1],
        # Malaria parameters
        prev=[0.25],
        Q0=[0.92],
        phi=[0.85],
        season=[1],
        routine=[0.1],
        # Current and future IRS
        irs=[0.0],
        irs_future=[0.3],
        # LSM coverage
        lsm=[0.0],
        # What to predict
        predictor=["prevalence"],
        # Enable benchmarking
        benchmark=True,
    )

    print("\nPrevalence results shape:", results.prevalence.shape)
    print("\nFirst few rows:")
    print(results.prevalence.head(10))

    return results


def example_multiple_scenarios():
    """Run multiple scenarios comparing different interventions."""
    print("\n" + "=" * 60)
    print("Example 2: Comparing Multiple Scenarios")
    print("=" * 60)

    # Define 4 scenarios with different interventions
    scenarios = {
        "Baseline": {
            "py_only": 0.5,
            "py_pbo": 0.0,
            "py_pyrrole": 0.0,
            "py_ppf": 0.0,
            "irs_future": 0.0,
        },
        "PBO_Nets": {
            "py_only": 0.0,
            "py_pbo": 0.5,
            "py_pyrrole": 0.0,
            "py_ppf": 0.0,
            "irs_future": 0.0,
        },
        "Mixed_Nets": {
            "py_only": 0.2,
            "py_pbo": 0.2,
            "py_pyrrole": 0.1,
            "py_ppf": 0.0,
            "irs_future": 0.0,
        },
        "Nets_Plus_IRS": {
            "py_only": 0.3,
            "py_pbo": 0.2,
            "py_pyrrole": 0.0,
            "py_ppf": 0.0,
            "irs_future": 0.4,
        },
    }

    n_scenarios = len(scenarios)
    tags = list(scenarios.keys())

    results = run_minter_scenarios(
        res_use=[0.4] * n_scenarios,
        py_only=[s["py_only"] for s in scenarios.values()],
        py_pbo=[s["py_pbo"] for s in scenarios.values()],
        py_pyrrole=[s["py_pyrrole"] for s in scenarios.values()],
        py_ppf=[s["py_ppf"] for s in scenarios.values()],
        prev=[0.30] * n_scenarios,
        Q0=[0.92] * n_scenarios,
        phi=[0.85] * n_scenarios,
        season=[1] * n_scenarios,
        routine=[0.1] * n_scenarios,
        irs=[0.0] * n_scenarios,
        irs_future=[s["irs_future"] for s in scenarios.values()],
        lsm=[0.0] * n_scenarios,
        scenario_tag=tags,
        predictor=["prevalence", "cases"],
        benchmark=True,
    )

    # Print summary by scenario
    print("\n--- Prevalence Summary (mean by scenario) ---")
    if results.prevalence is not None:
        summary = results.prevalence.groupby("scenario")["prevalence"].agg(
            ["mean", "min", "max"]
        )
        print(summary)

    return results


def example_direct_emulator():
    """Use the emulator directly with a scenarios DataFrame."""
    print("\n" + "=" * 60)
    print("Example 3: Direct Emulator Usage")
    print("=" * 60)

    # Create scenarios DataFrame directly
    scenarios = create_scenarios(
        eir=[50, 100, 150, 200],
        dn0_use=[0.5, 0.45, 0.4, 0.35],
        dn0_future=[0.6, 0.55, 0.5, 0.45],
        Q0=[0.92, 0.92, 0.92, 0.92],
        phi_bednets=[0.85, 0.85, 0.85, 0.85],
        seasonal=[1, 1, 1, 1],
        routine=[0.1, 0.1, 0.1, 0.1],
        itn_use=[0.6, 0.55, 0.5, 0.45],
        irs_use=[0.0, 0.0, 0.0, 0.0],
        itn_future=[0.7, 0.65, 0.6, 0.55],
        irs_future=[0.3, 0.3, 0.3, 0.3],
        lsm=[0.0, 0.1, 0.2, 0.3],
    )

    print("Scenarios DataFrame:")
    print(scenarios)

    # Run emulator
    results = run_malaria_emulator(
        scenarios=scenarios,
        predictor="prevalence",
        model_types=["LSTM"],
        time_steps=2190,  # 6 years
        benchmark=True,
    )

    print("\nResults shape:", results.shape)
    print("\nSample predictions:")
    print(results.head(20))

    return results


def example_batch_scenarios():
    """Run a large batch of random scenarios."""
    print("\n" + "=" * 60)
    print("Example 4: Batch Processing (100 scenarios)")
    print("=" * 60)

    np.random.seed(42)
    n_scenarios = 100

    results = run_minter_scenarios(
        res_use=np.random.uniform(0.1, 0.8, n_scenarios),
        py_only=np.random.uniform(0, 0.4, n_scenarios),
        py_pbo=np.random.uniform(0, 0.3, n_scenarios),
        py_pyrrole=np.random.uniform(0, 0.2, n_scenarios),
        py_ppf=np.random.uniform(0, 0.1, n_scenarios),
        prev=np.random.uniform(0.1, 0.5, n_scenarios),
        Q0=np.full(n_scenarios, 0.92),
        phi=np.full(n_scenarios, 0.85),
        season=np.ones(n_scenarios),
        routine=np.full(n_scenarios, 0.1),
        irs=np.zeros(n_scenarios),
        irs_future=np.random.uniform(0, 0.4, n_scenarios),
        lsm=np.random.uniform(0, 0.2, n_scenarios),
        scenario_tag=[f"Batch_{i:03d}" for i in range(n_scenarios)],
        predictor=["prevalence"],
        benchmark=True,
    )

    print(f"\nProcessed {n_scenarios} scenarios")
    print(f"Total rows in results: {len(results.prevalence)}")

    if results.benchmarks:
        print(f"\nPerformance: {results.benchmarks['total']:.2f} seconds total")
        print(f"  = {results.benchmarks['total'] / n_scenarios * 1000:.1f} ms per scenario")

    return results


def main():
    """Run all examples."""
    print("\n" + "#" * 60)
    print("# MINTe Package Examples")
    print("#" * 60)

    # Note: These examples require model files to be present
    # Without model files, they will raise FileNotFoundError

    try:
        # Preload models for efficiency
        print("\nPreloading models...")
        preload_all_models(verbose=True)
    except FileNotFoundError:
        print("\n[WARNING] Model files not found.")
        print("To run these examples, you need to:")
        print("1. Place model files in src/minte/models/")
        print("2. Or set MINTER_MODELS_DIR environment variable")
        print("\nSkipping examples that require models.")
        return

    # Run examples
    example_basic_scenario()
    example_multiple_scenarios()
    example_direct_emulator()
    example_batch_scenarios()

    print("\n" + "#" * 60)
    print("# All examples completed!")
    print("#" * 60)


if __name__ == "__main__":
    main()
