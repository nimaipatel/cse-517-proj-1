#!/usr/bin/env python3

import matplotlib.pyplot as plt

from typing import List

# pright: strict
# workaround for unknown matplotlib types:
# pyright: reportUnknownMemberType=false
# pyright: reportMissingImports=false

from queues import Simulation, Simulation_Run, Expected_Value_List, Expected_Value_Dist


def Plot_Stats(plt, means: List[float], margins_of_error: List[float], labels: List[str], title: str, metric: str, xlabel: str):  # type: ignore
    assert (len(means) == len(margins_of_error))
    x = range(len(means))

    plt.bar(x, means, yerr=margins_of_error, capsize=5, alpha=0.0)

    plt.plot(x, means, 'o', color='black', label='Mean')

    plt.set_xlabel(xlabel)
    plt.set_ylabel(f'{metric}')
    plt.set_title(title)
    plt.set_xticks(x, labels)
    plt.axhline(0, color='grey', linewidth=0.8)
    plt.legend()


def Plot_Varying():
    _, axes = plt.subplots(1, 2, figsize=(12, 10))  # type: ignore

    sojourn_times_means: List[float] = []
    sojourn_times_moes: List[float] = []

    num_jobs_means: List[float] = []
    num_jobs_moes: List[float] = []

    labels: List[str] = []

    mu1 = 8
    mu2 = 9
    for lam in range(1, 6):

        s = Simulation(
            DURATION=1000,
            ARRIVAL_RATE=lam,
            SERVICE_RATE_1=mu1,
            SERVICE_RATE_2=mu2,
        )

        _, _, sys_freq = Simulation_Run(s)
        avg_sojourn_time, _, moe_sojourn_time = Expected_Value_List(
            s.sojourn_times)
        avg_num_jobs, _, moe_num_jobs = Expected_Value_Dist(sys_freq)

        sojourn_times_means.append(avg_sojourn_time)
        sojourn_times_moes.append(moe_sojourn_time)

        num_jobs_means.append(avg_num_jobs)
        num_jobs_moes.append(moe_num_jobs)

        labels.append(f"{lam}")

    title = f"Service Rate 1 = {mu1}, Service Rate 2 = {mu2}, Varying Arrival Rate"
    Plot_Stats(axes[0],  # type: ignore
               sojourn_times_means, sojourn_times_moes,
               labels, title, "sojourn time (s)", "Arrival Rate")
    Plot_Stats(axes[1],  # type: ignore
               num_jobs_means, num_jobs_moes, labels, title, "number of jobs", "Arrival Rate")
    plt.tight_layout()
    plt.savefig('varying_arrival.png', format='png')
    print("Generated varying_arrival.png...")


def Plot_Varying_Service_1():
    _, axes = plt.subplots(1, 2, figsize=(12, 10))  # type: ignore

    sojourn_times_means: List[float] = []
    sojourn_times_moes: List[float] = []

    num_jobs_means: List[float] = []
    num_jobs_moes: List[float] = []

    labels: List[str] = []

    lam = 1
    mu2 = 3
    for mu1 in range(3, 13):

        s = Simulation(
            DURATION=1000,
            ARRIVAL_RATE=lam,
            SERVICE_RATE_1=mu1,
            SERVICE_RATE_2=mu2,
        )

        _, _, sys_freq = Simulation_Run(s)

        avg_sojourn_time, _, moe_sojourn_time = Expected_Value_List(
            s.sojourn_times)
        avg_num_jobs, _, moe_num_jobs = Expected_Value_Dist(sys_freq)

        sojourn_times_means.append(avg_sojourn_time)
        sojourn_times_moes.append(moe_sojourn_time)

        num_jobs_means.append(avg_num_jobs)
        num_jobs_moes.append(moe_num_jobs)

        labels.append(f"{mu1}")

    title = f"Arrival Rate = {lam}, Service Rate 2 = {mu2}, Varying Service Rate 1"
    Plot_Stats(axes[0],  # type: ignore
               sojourn_times_means, sojourn_times_moes, labels, title, "sojourn time (s)", "Service Rate 1")
    Plot_Stats(axes[1],  # type: ignore
               num_jobs_means, num_jobs_moes, labels, title, "number of jobs", "Service Rate 1")

    plt.tight_layout()
    plt.savefig('varying_1.png', format='png')
    print("Generated varying_1.png...")


def Plot_Varying_Service_2():
    _, axes = plt.subplots(1, 2, figsize=(12, 10))  # type: ignore

    sojourn_times_means: List[float] = []
    sojourn_times_moes: List[float] = []

    num_jobs_means: List[float] = []
    num_jobs_moes: List[float] = []

    labels: List[str] = []

    lam = 1
    mu1 = 3
    for mu2 in range(3, 13):

        s = Simulation(
            DURATION=1000,
            ARRIVAL_RATE=lam,
            SERVICE_RATE_1=mu1,
            SERVICE_RATE_2=mu2,
        )

        _, _, sys_freq = Simulation_Run(s)

        avg_sojourn_time, _, moe_sojourn_time = Expected_Value_List(
            s.sojourn_times)
        avg_num_jobs, _, moe_num_jobs = Expected_Value_Dist(sys_freq)

        sojourn_times_means.append(avg_sojourn_time)
        sojourn_times_moes.append(moe_sojourn_time)

        num_jobs_means.append(avg_num_jobs)
        num_jobs_moes.append(moe_num_jobs)

        labels.append(f"{mu2}")

    title = f"Arrival Rate = {lam}, Service Rate 1 = {mu1}, Varying Service Rate 2"

    Plot_Stats(axes[0],  # type: ignore
               sojourn_times_means, sojourn_times_moes, labels, title, "sojourn time (s)", "Service Rate 2")
    Plot_Stats(axes[1],  # type: ignore
               num_jobs_means, num_jobs_moes, labels, title, "number of jobs", "Service Rate 2")

    plt.tight_layout()
    plt.savefig('varying_2.png', format='png')
    print("Generated varying_2.png...")


def Plot_Varying_Service_Both():
    _, axes = plt.subplots(1, 2, figsize=(12, 10))  # type: ignore

    sojourn_times_means: List[float] = []
    sojourn_times_moes: List[float] = []

    num_jobs_means: List[float] = []
    num_jobs_moes: List[float] = []

    labels: List[str] = []

    lam = 1
    for mu in range(3, 13):
        mu1 = mu
        mu2 = mu

        s = Simulation(
            DURATION=1000,
            ARRIVAL_RATE=lam,
            SERVICE_RATE_1=mu1,
            SERVICE_RATE_2=mu2,
        )

        _, _, sys_freq = Simulation_Run(s)

        avg_sojourn_time, _, moe_sojourn_time = Expected_Value_List(
            s.sojourn_times)
        avg_num_jobs, _, moe_num_jobs = Expected_Value_Dist(sys_freq)

        sojourn_times_means.append(avg_sojourn_time)
        sojourn_times_moes.append(moe_sojourn_time)

        num_jobs_means.append(avg_num_jobs)
        num_jobs_moes.append(moe_num_jobs)

        labels.append(f"{mu}")

    title = f"Arrival Rate = {lam}, Varying Service Rate 1 and 2"

    Plot_Stats(axes[0],  # type: ignore
               sojourn_times_means, sojourn_times_moes, labels, title, "sojourn time (s)", "Service Rate 1 and 2")
    Plot_Stats(axes[1],  # type: ignore
               num_jobs_means, num_jobs_moes, labels, title, "number of jobs", "Service Rate 1 and 2")

    plt.tight_layout()
    plt.savefig('varying_both.png', format='png')
    print("Generated varying_both.png...")


def Plot_Charts():
    Plot_Varying()
    Plot_Varying_Service_1()
    Plot_Varying_Service_2()
    Plot_Varying_Service_Both()
