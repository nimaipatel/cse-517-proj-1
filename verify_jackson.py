#!/usr/bin/env python3

# Copyright (C) 2024 Patel, Nimai <nimai.m.patel@gmail.com>
# Author: Patel, Nimai <nimai.m.patel@gmail.com>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

# pyright: strict

# This program has been developed and tested on Python 3.9.6

import random
from queues import (
    Dist_Print,
    Expected_Value_Dist,
    Expected_Value_List,
    Exponential_Random,
    Freq_To_Prob,
    Simulation,
    Simulation_Run,
)


def main():
    random.seed(0)

    arrival_rate = 2
    service_rate_1 = 9
    service_rate_2 = 10
    LOG_FILE_NAME = "event_log.csv"

    s = Simulation(
        DURATION=1000,
        ARRIVAL_DIST=lambda: Exponential_Random(arrival_rate),
        SERVICE_1_DIST=lambda: Exponential_Random(service_rate_1),
        SERVICE_2_DIST=lambda: Exponential_Random(service_rate_2),
        LOG_FILE_NAME=LOG_FILE_NAME,
    )

    q1_freq, q2_freq, sys_freq, sojourn_times = Simulation_Run(s)

    q1_probs = Freq_To_Prob(q1_freq)
    q2_probs = Freq_To_Prob(q2_freq)
    sys_probs = Freq_To_Prob(sys_freq)

    avg_sojourn_time, sd_sojourn_time, moe_sojourn_time = Expected_Value_List(
        sojourn_times
    )
    avg_num_jobs, sd_num_jobs, moe_num_jobs = Expected_Value_Dist(sys_freq)

    avg_sojourn_time_exp = 1 / (service_rate_1 - arrival_rate)
    avg_sojourn_time_exp += 1 / (service_rate_2 - arrival_rate)
    avg_num_jobs_exp = arrival_rate * avg_sojourn_time_exp

    q1_jacksons, q2_jacksons, q3_jacksons = Verify_Jacksons_Theorem(
        arrival_rate, service_rate_1, service_rate_2, 10
    )

    print("Probability Distribution from simualtion:")
    Dist_Print(q1_probs, q2_probs, sys_probs)

    print()

    print("Equilibrium State Probability Distribution from Jackson's Theorem:")
    Dist_Print(q1_jacksons, q2_jacksons, q3_jacksons)

    print()

    print(f"Total jobs inbound   = {s.total_jobs}")
    print(f"Total jobs completed = {s.comp_jobs}")
    print()
    print(f"Average Sojourn Time (expected) = {avg_sojourn_time_exp:.5f}s")
    print(f"Average Sojourn Time (measured) = {avg_sojourn_time:.5f}s")
    print(f"Standard Deviation              = {sd_sojourn_time:.5f}s")
    print(
        f"95% Confidence interval         = ({avg_sojourn_time:.5f} ± {moe_sojourn_time:.5f})s"
    )
    print()
    print(f"Average number of jobs in system (expected) = {avg_num_jobs_exp:.5f}")
    print(f"Average number of jobs in system (measured) = {avg_num_jobs:.5f}")
    print(f"Standard Deviation                          = {sd_num_jobs:.5f}")
    print(
        f"95% Confidence interval                     = ({avg_num_jobs:.5f} ± {moe_num_jobs:.5f})s"
    )
    print()

    print(f"Refer {LOG_FILE_NAME} for event log")


def Verify_Jacksons_Theorem(
    arrival_rate: float,
    service_rate_1: float,
    service_rate_2: float,
    n: int,
) -> tuple[dict[int, float], dict[int, float], dict[int, float]]:

    q1_cal_probs: dict[int, float] = {}
    rho1 = arrival_rate / service_rate_1
    q1_cal_probs[0] = 1 - rho1
    for k in range(1, n):
        q1_cal_probs[k] = q1_cal_probs[k - 1] * rho1

    q2_cal_probs: dict[int, float] = {}
    rho2 = arrival_rate / service_rate_2
    q2_cal_probs[0] = 1 - rho2
    for k in range(1, n):
        q2_cal_probs[k] = q2_cal_probs[k - 1] * rho2

    sys_cal_probs: dict[int, float] = {}
    for sys_len in range(n):
        sys_cal_probs[sys_len] = 0
        for i in range(0, sys_len + 1):
            q1_len = i
            q2_len = sys_len - i

            if q1_len in q1_cal_probs and q2_len in q2_cal_probs:
                prob = q1_cal_probs[q1_len] * q2_cal_probs[q2_len]
                sys_cal_probs[sys_len] += prob

    return q1_cal_probs, q2_cal_probs, sys_cal_probs


if __name__ == "__main__":
    main()
