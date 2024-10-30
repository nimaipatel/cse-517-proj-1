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
import itertools
import math
import random
from enum import Enum, auto
from typing import List, NewType, Optional, Callable

JobID = NewType("JobID", int)
PhaseDist = tuple[list[float], list[list[float]]]


class EventType(Enum):
    ARRIVAL = auto()
    START = auto()
    COMPLETE = auto()


@dataclass
class Event:
    time: float
    type: EventType
    job_id: JobID
    queue_index: int


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
class QueueState:
    q: List[JobID] = field(default_factory=list)
    busy = False


@dataclass
class Simulation:
    # Constants
    DURATION: float
    # yeah, function pointers are slow, but we are already using Python...
    ARRIVAL_DIST: Callable[[], float]
    SERVICE_DIST: list[Callable[[], float]]
    LOG_FILE_NAME: Optional[str]

    # State
    clock: float = 0
    queue_states: list[QueueState] = field(default_factory=list)
    es = EventStack()

    # ...state for performance metrics
    total_jobs = 0
    comp_jobs = 0
    sojourn_times: List[float] = field(default_factory=list)
    arrival_times: dict[JobID, float] = field(default_factory=dict)

    def __post_init__(self):
        self.queue_states = [QueueState() for _ in range(len(self.SERVICE_DIST))]


def Event_Log_Open(s: Simulation):
    if s.LOG_FILE_NAME is None:
        return

    with open(s.LOG_FILE_NAME, "w") as f:
        f.write("Job ID, Event Type, Event Time\n")


def Event_Log_Record(
    s: Simulation, queue_index: int, job_id: JobID, event: EventType, time: float
) -> None:
    if s.LOG_FILE_NAME is None:
        return

    with open(s.LOG_FILE_NAME, "a") as f:
        event_str = event.name
        if event != EventType.ARRIVAL:
            event_str += f" {queue_index}"
        f.write(f"{job_id}, {event_str}, {time}\n")


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

    Event_Log_Record(s, 0, job_id, EventType.ARRIVAL, s.clock)

    if s.queue_states[0].busy:
        s.queue_states[0].q.append(job_id)
    else:
        Start_Service(s, 0, job_id)

    inter_arrival_time = s.ARRIVAL_DIST()
    next_arrival_event = Event(
        s.clock + inter_arrival_time, EventType.ARRIVAL, job_id, 0
    )
    Event_Stack_Insert(s.es, next_arrival_event)


def Start_Service(s: Simulation, queue_index: int, job_id: JobID):
    Event_Log_Record(s, queue_index, job_id, EventType.START, s.clock)

    s.queue_states[queue_index].busy = True

    serv_time = s.SERVICE_DIST[queue_index]()
    event = Event(s.clock + serv_time, EventType.COMPLETE, job_id, queue_index)
    Event_Stack_Insert(s.es, event)


def Complete_Service(s: Simulation, queue_index: int, job_id: JobID):
    Event_Log_Record(s, queue_index, job_id, EventType.COMPLETE, s.clock)

    s.queue_states[queue_index].busy = False

    next_queue_index = queue_index + 1
    if next_queue_index < len(s.queue_states):
        if s.queue_states[next_queue_index].busy:
            s.queue_states[next_queue_index].q.append(job_id)
        else:
            Start_Service(s, next_queue_index, job_id)
    else:
        s.sojourn_times.append(s.clock - s.arrival_times[job_id])
        s.comp_jobs += 1

    if len(s.queue_states[queue_index].q) > 0:
        next_job_id = s.queue_states[queue_index].q.pop(0)
        Start_Service(s, queue_index, next_job_id)


def Exponential_Random(rate: float) -> float:
    U = random.random()
    return -math.log(1 - U) / rate


def Phase_Type_Random(alpha: list[float], rates: list[float], T: list[list[float]]):
    state = random.choices(range(len(alpha)), weights=alpha)[0]

    time = 0.0
    while state < len(rates):
        time += Exponential_Random(rates[state])
        state = random.choices(range(len(T[state])), weights=T[state])[0]

    return time


def Get_Phase_Type_Dist(
    alpha: list[float], S: list[list[float]]
) -> Callable[[], float]:
    N_states = len(alpha)

    assert len(S) == N_states
    for state in range(len(S)):
        assert len(S[state]) == N_states
        assert S[state][state] <= 0
        assert sum(S[state]) <= 0

    rates = [-S[state][state] for state in range(len(S))]

    T = [[0.0] * (len(S) + 1) for _ in range(len(S))]

    for state, next_state in itertools.product(range(len(S)), repeat=2):
        # every transient state to every other transient state...
        if state != next_state:
            T[state][next_state] = S[state][next_state]

    for state in range(len(S)):
        # every transient state to the single abosorbing state...
        T[state][len(S)] = -sum(S[state])

    return lambda: Phase_Type_Random(alpha, rates, T)


def Get_Erlang_Dist(k: int, lam: float) -> Callable[[], float]:
    alpha = [1.0] + [0.0] * (k - 1)

    S: list[list[float]] = []
    for i in range(k):
        row = [0.0 for _ in range(k)]
        row[i] = -lam
        if i + 1 < len(row):
            row[i + 1] = lam
        S.append(row)

    return Get_Phase_Type_Dist(alpha, S)


def Get_Exponential_Dist(lam: float) -> Callable[[], float]:
    return Get_Phase_Type_Dist([1.0], [[-lam]])


def Simulation_Run(s: Simulation):
    qs_freq: list[defaultdict[int, float]] = [
        defaultdict(lambda: 0) for _ in range(len(s.queue_states))
    ]
    sys_freq: defaultdict[int, float] = defaultdict(lambda: 0)

    Event_Log_Open(s)

    Arrival(s)
    while (
        (not Event_Stack_Is_Empty(s.es))
        and (event := Event_Stack_Pop(s.es))
        and (event.time < s.DURATION)
    ):

        elapsed = event.time - s.clock

        total_n = 0
        for i in range(len(s.queue_states)):
            n = len(s.queue_states[i].q)
            if s.queue_states[i].busy:
                n += 1

            total_n += n

            qs_freq[i][n] += elapsed

        sys_freq[total_n] += elapsed

        s.clock = event.time

        if event.type == EventType.ARRIVAL:
            Arrival(s)
        elif event.type == EventType.COMPLETE:
            Complete_Service(s, event.queue_index, event.job_id)
        else:
            assert False, "UNREACHABLE"

    for i in range(len(s.queue_states)):
        assert max(qs_freq[i].keys()) == len(qs_freq[i].keys()) - 1
    assert max(sys_freq.keys()) == len(sys_freq.keys()) - 1

    return qs_freq, sys_freq, s.sojourn_times


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
