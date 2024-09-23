# Copyright (C) 2024 Patel, Nimai <nimai.m.patel@gmail.com>
# Author: Patel, Nimai <nimai.m.patel@gmail.com>

from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass, field
import math
import random
from enum import Enum, auto
import sys
from typing import List, NewType, Optional
import argparse

job_id_t = NewType("job_id_t", int)


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
    job_id: job_id_t


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
    ARRIVAL_RATE: float
    SERVICE_RATE_1: float
    SERVICE_RATE_2: float

    # State
    clock: float = 0
    q1: List[job_id_t] = field(default_factory=list)
    q2: List[job_id_t] = field(default_factory=list)
    server_1_busy = False
    server_2_busy = False
    es = EventStack()

    # ...state for performance metrics
    total_jobs = 0
    comp_jobs = 0
    total_sojourn_time: float = 0
    arrival_times: dict[job_id_t, float] = field(default_factory=dict)


# Misc.
event_log_file_handle = None


def open_event_log_file(sim_id: str):
    global event_log_file_handle

    file_name = f"event-log-{sim_id}.csv"
    event_log_file_handle = open(file_name, "w")
    event_log_file_handle.truncate(0)
    event_log_file_handle.write("Job ID, Event Type, Event Time\n")

    return file_name


def log_event(job_id: job_id_t, event: EventType, time: float):
    global event_log_file_handle

    assert event_log_file_handle != None
    event_log_file_handle.write(f"{job_id}, {event.name}, {time}\n")


def close_event_log_file():
    global event_log_file_handle

    assert event_log_file_handle != None
    event_log_file_handle.close()


def exponential_random(mean: float) -> float:
    U = random.random()
    return -math.log(1 - U) * mean


def ES_Is_Empty(es: EventStack):
    return es.head == None


def ES_Insert(es: EventStack, event: Event):
    node = EventStackNode(event)

    if ES_Is_Empty(es):
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

    if curr.prev:
        curr.prev.next = node
    curr.prev = node


def ES_Pop(es: EventStack):
    assert es.head != None
    event = es.head.event
    es.head = es.head.next

    if es.head:
        es.head.prev = None
    else:
        es.tail = None

    return event


def Arrival(s: Simulation):
    s.total_jobs += 1
    job_id = job_id_t(s.total_jobs)
    s.arrival_times[job_id] = s.clock

    log_event(job_id, EventType.ARRIVAL, s.clock)

    if s.server_1_busy:
        s.q1.append(job_id)
    else:
        Start_Service_1(s, job_id)

    inter_arrival_time = exponential_random(1 / s.ARRIVAL_RATE)
    next_arrival_event = Event(s.clock + inter_arrival_time, EventType.ARRIVAL, job_id)
    ES_Insert(s.es, next_arrival_event)


def Start_Service_1(s: Simulation, job_id: job_id_t):
    log_event(job_id, EventType.START_SERVICE_1, s.clock)

    s.server_1_busy = True

    serv_time = exponential_random(1 / s.SERVICE_RATE_1)
    event = Event(s.clock + serv_time, EventType.COMPLETE_SERVICE_1, job_id)
    ES_Insert(s.es, event)


def Complete_Service_1(s: Simulation, job_id: job_id_t):
    log_event(job_id, EventType.COMPLETE_SERVICE_1, s.clock)

    s.server_1_busy = False

    if s.server_2_busy:
        s.q2.append(job_id)
    else:
        Start_Service_2(s, job_id)

    if len(s.q1) > 0:
        next_job_id = s.q1.pop(0)
        Start_Service_1(s, next_job_id)


def Start_Service_2(s: Simulation, job_id: job_id_t):
    log_event(job_id, EventType.START_SERVICE_2, s.clock)

    s.server_2_busy = True

    serv_time = exponential_random(1 / s.SERVICE_RATE_2)
    event = Event(s.clock + serv_time, EventType.COMPLETE_SERVICE_2, job_id)
    ES_Insert(s.es, event)


def Complete_Service_2(s: Simulation, job_id: job_id_t):
    log_event(job_id, EventType.COMPLETE_SERVICE_2, s.clock)
    s.server_2_busy = False

    s.total_sojourn_time += s.clock - s.arrival_times.pop(job_id)
    s.comp_jobs += 1

    if len(s.q2) > 0:
        next_job_id = s.q2.pop(0)
        Start_Service_2(s, next_job_id)


def Simulation_Run(
    s: Simulation,
) -> tuple[dict[int, float], dict[int, float], dict[int, float]]:
    """Runs the simulation and returns probability distributions for queue 1, queue 2 and overall system"""
    q1_len = 0
    q2_len = 0

    q1_freq: defaultdict[int, float] = defaultdict(lambda: 0)
    q2_freq: defaultdict[int, float] = defaultdict(lambda: 0)
    overall_freq: defaultdict[int, float] = defaultdict(lambda: 0)

    Arrival(s)
    while not ES_Is_Empty(s.es) and s.clock < s.DURATION:
        event = ES_Pop(s.es)

        elapsed = event.time - s.clock
        q1_freq[q1_len] += elapsed
        q2_freq[q2_len] += elapsed
        overall_freq[q1_len + q2_len] += elapsed

        s.clock = event.time
        q1_len = len(s.q1)
        q2_len = len(s.q2)

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
    assert max(overall_freq.keys()) == len(overall_freq.keys()) - 1

    return q1_freq, q2_freq, overall_freq


def Get_Config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=float)
    parser.add_argument("--arrival-rate", type=float)
    parser.add_argument("--service-rate-1", type=float)
    parser.add_argument("--service-rate-2", type=float)
    parser.add_argument("--rng-seed", type=int)

    args = parser.parse_args()

    if args.duration == None:
        print("Duration not provided, using default value")
        args.duration = 10_000
    if args.arrival_rate == None:
        print("Arrival rate not provided, using default value")
        args.arrival_rate = 1
    if args.service_rate_1 == None:
        print("Queue 1 service rate not provided, using default value")
        args.service_rate_1 = 3
    if args.service_rate_2 == None:
        print("Queue 2 service rate not provided, using default value")
        args.service_rate_2 = 4
    if args.rng_seed == None:
        print("Seed for RNG not provided, using random seed")
        args.rng_seed = random.randrange(sys.maxsize)

    print()

    print(f"Duration\t= {args.duration} seconds")
    print(f"Arrival Rate\t= {args.arrival_rate} jobs/second")
    print(f"Service Rate 1\t= {args.service_rate_1} jobs/second")
    print(f"Service Rate 2\t= {args.service_rate_2} jobs/second")
    print(f"RNG Seed\t= {args.rng_seed}")

    print()

    return (
        args.duration,
        args.arrival_rate,
        args.service_rate_1,
        args.service_rate_2,
        args.rng_seed,
    )


def main():
    duration, arrival_rate, service_rate_1, service_rate_2, rng_seed = Get_Config()

    sim_id = f"{duration}-{arrival_rate}-{service_rate_1}-{service_rate_2}-{rng_seed}"

    event_log_file_name = open_event_log_file(sim_id)

    s = Simulation(
        DURATION=duration,
        ARRIVAL_RATE=arrival_rate,
        SERVICE_RATE_1=service_rate_1,
        SERVICE_RATE_2=service_rate_2,
    )

    random.seed(rng_seed)

    q1_freq, q2_freq, overall_freq = Simulation_Run(s)

    q1_probs = freq_to_prob(q1_freq)
    q2_probs = freq_to_prob(q2_freq)
    overall_probs = freq_to_prob(overall_freq)

    verify_jacksons_theorem(q1_probs, q2_probs, overall_probs)

    print()

    avg_sojourn_time = s.total_sojourn_time / s.comp_jobs
    print(f"Total jobs inbound: {s.total_jobs}")
    print(f"Total jobs completed: {s.comp_jobs}")
    print(f"Measured Sojourn Time\t= {avg_sojourn_time} seconds")
    print(
        f"Expected Sojourn Time\t= {1 / (s.SERVICE_RATE_1 - s.ARRIVAL_RATE) + 1 / (s.SERVICE_RATE_2 - s.ARRIVAL_RATE)}"
    )
    print(f"Avg number of jobs (measured) = {expected_value(overall_probs)}")

    print()

    print(f"Refer {event_log_file_name} for event log")

    close_event_log_file()


def freq_to_prob(freq_dist: dict[int, float]) -> dict[int, float]:
    """Takes dictionary that stores frequency distribution and returns
    dictionary that stores probability distribution"""
    prob_dist: dict[int, float] = {}
    total_freq = 0
    for freq in freq_dist.values():
        total_freq += freq

    for var, freq in freq_dist.items():
        prob_dist[var] = freq_dist[var] / total_freq

    return prob_dist


def expected_value(prob_dist: dict[int, float]) -> float:
    exp_val = 0
    for var, prob in prob_dist.items():
        exp_val += var * prob
    return exp_val


def verify_jacksons_theorem(
    q1_probs: dict[int, float],
    q2_probs: dict[int, float],
    measu_overall_probs: dict[int, float],
):
    calcu_overall_probs: dict[int, float] = {}
    for overall_len in measu_overall_probs.keys():
        calcu_overall_probs[overall_len] = 0
        for i in range(0, overall_len + 1):
            q1_len = i
            q2_len = overall_len - i

            if q1_len in q1_probs and q2_len in q2_probs:
                calcu_overall_probs[overall_len] += q1_probs[q1_len] * q2_probs[q2_len]

    print(
        "Size",
        "Queue 1",
        "Queue 2",
        "Overall",
        "Overall (calculated)",
        sep="\t",
    )

    most_len = max(
        len(q1_probs), len(q2_probs), len(measu_overall_probs), len(calcu_overall_probs)
    )
    for i in range(most_len + 1):
        print(
            i,
            "{:.5f}".format(q1_probs.get(i, 0)),
            "{:.5f}".format(q2_probs.get(i, 0)),
            "{:.5f}".format(measu_overall_probs.get(i, 0)),
            "{:.5f}".format(calcu_overall_probs.get(i, 0)),
            sep="\t",
        )


if __name__ == "__main__":
    main()
