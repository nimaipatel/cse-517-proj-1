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

from queues import (
    Dist_Print,
    Expected_Value_Dist,
    Expected_Value_List,
    Freq_To_Prob,
    Get_Erlang_Dist,
    Get_Exponential_Dist,
    Simulation,
    Simulation_Run,
)

s = Simulation(
    DURATION=1000,
    ARRIVAL_DIST=Get_Erlang_Dist(2, 3),
    SERVICE_1_DIST=Get_Exponential_Dist(10),
    SERVICE_2_DIST=Get_Exponential_Dist(11),
    LOG_FILE_NAME=None,
)

q1_freq, q2_freq, sys_freq, sojourn_times = Simulation_Run(s)

q1_probs = Freq_To_Prob(q1_freq)
q2_probs = Freq_To_Prob(q2_freq)
sys_probs = Freq_To_Prob(sys_freq)

avg_sojourn_time, sd_sojourn_time, moe_sojourn_time = Expected_Value_List(sojourn_times)
avg_num_jobs, sd_num_jobs, moe_num_jobs = Expected_Value_Dist(sys_freq)


Dist_Print(q1_probs, q2_probs, sys_probs)
print(avg_sojourn_time)
print(avg_num_jobs)