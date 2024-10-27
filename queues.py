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

from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass, field
import math
import random
from enum import Enum, auto
import sys
from typing import List, NewType, Optional, Union, Callable
import argparse
import ast

JobID = NewType("JobID", int)
PhaseDist = Union[float, tuple[list[float], list[list[float]]]]


class EventType(Enum):
    ARRIVAL = auto()
    START_SERVICE_1 = auto()
    COMPLETE_SERVICE_1 = auto()
    START_SERVICE_2 = auto()
    COMPLETE_SERVICE_2 = auto()


@dataclass
class Event:
    time: float
    type: EventType
    job_id: JobID


@dataclass
class EventStackNode:
    event: Event
    next: Optional[EventStackNode] = None
    prev: Optional[EventStackNode] = None


@dataclass
class EventStack:
    head: Optional[EventStackNode] = None
    tail: Optional[EventStackNode] = None


@dataclass
class Simulation:
    # Constants
    DURATION: float
    ARRIVAL_DIST: Callable[
        [], float
    ]  # yeah, function pointers are slow, but guess what? we are using Python
    SERVICE_1_DIST: Callable[[], float]
    SERVICE_2_DIST: Callable[[], float]
    LOG_FILE_NAME: Optional[str]

    # State
    clock: float = 0
    q1: List[JobID] = field(default_factory=list)
    q2: List[JobID] = field(default_factory=list)
    server_1_busy = False
    server_2_busy = False
    es = EventStack()

    # ...state for performance metrics
    total_jobs = 0
    comp_jobs = 0
    sojourn_times: List[float] = field(default_factory=list)
    arrival_times: dict[JobID, float] = field(default_factory=dict)


def Event_Log_Open(s: Simulation):
    if s.LOG_FILE_NAME is None:
        return

    with open(s.LOG_FILE_NAME, "w") as f:
        f.write("Job ID, Event Type, Event Time\n")


def Event_Log_Record(
    s: Simulation, job_id: JobID, event: EventType, time: float
) -> None:
    if s.LOG_FILE_NAME is None:
        return

    with open(s.LOG_FILE_NAME, "a") as f:
        f.write(f"{job_id}, {event.name}, {time}\n")


def Event_Stack_Is_Empty(es: EventStack):
    return es.head == None


def Event_Stack_Insert(es: EventStack, event: Event):
    node = EventStackNode(event)

    if Event_Stack_Is_Empty(es):
        es.head = node
        es.tail = node
        return

    assert (
        es.head != None
    ), "head cannot be None, we checked that the event stack is not empty"
    if node.event.time <= es.head.event.time:
        node.next = es.head
        es.head.prev = node
        es.head = node
        return

    assert (
        es.tail != None
    ), "tail cannot be None, we checked that the event stack is not empty"
    if node.event.time >= es.tail.event.time:
        node.prev = es.tail
        es.tail.next = node
        es.tail = node
        return

    curr = es.head
    while curr and curr.event.time < node.event.time:
        curr = curr.next

    assert (
        curr != None
    ), "curr can't be None, insertion at end of the event stack is handled in O(1) already"
    node.next = curr
    node.prev = curr.prev

    assert curr.prev != None
    curr.prev.next = node
    curr.prev = node


def Event_Stack_Pop(es: EventStack):
    assert es.head != None, "Caller tried popping from empty event stack"
    event = es.head.event
    es.head = es.head.next

    if es.head:
        es.head.prev = None
    else:
        es.tail = None

    return event


def Arrival(s: Simulation):
    s.total_jobs += 1
    job_id = JobID(s.total_jobs)
    s.arrival_times[job_id] = s.clock

    Event_Log_Record(s, job_id, EventType.ARRIVAL, s.clock)

    if s.server_1_busy:
        s.q1.append(job_id)
    else:
        Start_Service_1(s, job_id)

    inter_arrival_time = s.ARRIVAL_DIST()
    next_arrival_event = Event(s.clock + inter_arrival_time, EventType.ARRIVAL, job_id)
    Event_Stack_Insert(s.es, next_arrival_event)


def Start_Service_1(s: Simulation, job_id: JobID):
    Event_Log_Record(s, job_id, EventType.START_SERVICE_1, s.clock)

    s.server_1_busy = True

    serv_time = s.SERVICE_1_DIST()
    event = Event(s.clock + serv_time, EventType.COMPLETE_SERVICE_1, job_id)
    Event_Stack_Insert(s.es, event)


def Complete_Service_1(s: Simulation, job_id: JobID):
    Event_Log_Record(s, job_id, EventType.COMPLETE_SERVICE_1, s.clock)

    s.server_1_busy = False

    if s.server_2_busy:
        s.q2.append(job_id)
    else:
        Start_Service_2(s, job_id)

    if len(s.q1) > 0:
        next_job_id = s.q1.pop(0)
        Start_Service_1(s, next_job_id)


def Start_Service_2(s: Simulation, job_id: JobID):
    Event_Log_Record(s, job_id, EventType.START_SERVICE_2, s.clock)
    s.server_2_busy = True

    serv_time = s.SERVICE_2_DIST()
    event = Event(s.clock + serv_time, EventType.COMPLETE_SERVICE_2, job_id)
    Event_Stack_Insert(s.es, event)


def Complete_Service_2(s: Simulation, job_id: JobID):
    Event_Log_Record(s, job_id, EventType.COMPLETE_SERVICE_2, s.clock)
    s.server_2_busy = False

    s.sojourn_times.append(s.clock - s.arrival_times[job_id])

    s.comp_jobs += 1

    if len(s.q2) > 0:
        next_job_id = s.q2.pop(0)
        Start_Service_2(s, next_job_id)


def Exponential_Random(rate: float) -> float:
    U = random.random()
    return -math.log(1 - U) / rate


def Simulation_Run(
    s: Simulation,
) -> tuple[dict[int, float], dict[int, float], dict[int, float], list[float]]:
    q1_freq: defaultdict[int, float] = defaultdict(lambda: 0)
    q2_freq: defaultdict[int, float] = defaultdict(lambda: 0)
    sys_freq: defaultdict[int, float] = defaultdict(lambda: 0)

    Event_Log_Open(s)

    Arrival(s)
    while (
        (not Event_Stack_Is_Empty(s.es))
        and (event := Event_Stack_Pop(s.es))
        and (event.time < s.DURATION)
    ):

        elapsed = event.time - s.clock

        n1 = len(s.q1)
        if s.server_1_busy:
            n1 += 1

        n2 = len(s.q2)
        if s.server_2_busy:
            n2 += 1

        q1_freq[n1] += elapsed
        q2_freq[n2] += elapsed
        sys_freq[n1 + n2] += elapsed

        s.clock = event.time

        if event.type == EventType.ARRIVAL:
            Arrival(s)
        elif event.type == EventType.COMPLETE_SERVICE_1:
            Complete_Service_1(s, event.job_id)
        elif event.type == EventType.COMPLETE_SERVICE_2:
            Complete_Service_2(s, event.job_id)
        else:
            assert False, "UNREACHABLE"

    assert max(q1_freq.keys()) == len(q1_freq.keys()) - 1
    assert max(q2_freq.keys()) == len(q2_freq.keys()) - 1
    assert max(sys_freq.keys()) == len(sys_freq.keys()) - 1

    return q1_freq, q2_freq, sys_freq, s.sojourn_times


def Parse_Phase_Dist(s: str) -> Optional[PhaseDist]:
    try:
        return float(s)
    except:
        try:
            d = ast.literal_eval(s)

            alpha = d[0]
            trans = d[1]

            return alpha, trans
        except:
            return None


def Get_Config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=float, default=10_000)
    parser.add_argument("--arrival-rate", type=float, default=2)
    parser.add_argument("--service-rate-1", type=float, default=9)
    parser.add_argument("--service-rate-2", type=float, default=10)
    parser.add_argument("--rng-seed", type=int, default=random.randrange(sys.maxsize))

    args = parser.parse_args()

    print(f"Print help using `{sys.argv[0]} -h` to modify these values...")
    print(f"Duration       = {args.duration}s")
    print(f"Arrival Rate   = {args.arrival_rate} job/s")
    print(f"Service Rate 1 = {args.service_rate_1} job/s")
    print(f"Service Rate 2 = {args.service_rate_2} job/s")
    print(f"RNG Seed       = {args.rng_seed}")
    print()

    return (
        args.duration,
        args.arrival_rate,
        args.service_rate_1,
        args.service_rate_2,
        args.rng_seed,
    )


def Freq_To_Prob(freq_dist: dict[int, float]) -> dict[int, float]:
    prob_dist: dict[int, float] = {}
    total_freq = 0
    for freq in freq_dist.values():
        total_freq += freq

    for var, freq in freq_dist.items():
        prob_dist[var] = freq_dist[var] / total_freq

    return prob_dist


def Expected_Value_List(l: list[float]) -> tuple[float, float, float]:
    mean = sum(l) / len(l)

    variance: float = sum((x - mean) ** 2 for x in l)
    variance /= len(l)

    sd = variance**0.5

    zstar = 1.96  # large distribution

    moe = zstar * sd / (len(l) ** 0.5)

    return mean, sd, moe


def Expected_Value_Dist(dist: dict[int, float]) -> tuple[float, float, float]:
    total_freq = sum(dist.values())

    mean = sum(x * freq for x, freq in dist.items())
    mean /= total_freq

    variance = sum(freq * (x - mean) ** 2 for x, freq in dist.items())
    variance /= total_freq

    sd = variance**0.5

    zstar = 1.96  # large distribution

    moe = zstar * sd / (total_freq**0.5)

    return mean, sd, moe


def Dist_Print(
    q1_probs: dict[int, float],
    q2_probs: dict[int, float],
    sys_probs: dict[int, float],
):
    most_len = max(len(q1_probs), len(q2_probs), len(sys_probs)) + 1

    format_string = "| {:<4} | {:<8} | {:<8} | {:<8} |"

    print(
        format_string.format(
            "Size",
            "Queue 1",
            "Queue 2",
            "System",
        )
    )

    for i in range(most_len):
        print(
            format_string.format(
                i,
                "{:.5f}".format(q1_probs.get(i, 0)),
                "{:.5f}".format(q2_probs.get(i, 0)),
                "{:.5f}".format(sys_probs.get(i, 0)),
            )
        )

    print("|_______________________________________|")
